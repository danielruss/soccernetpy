use pyo3::prelude::*;

/// Python bindings for SOCcerNET occupational coding and CLIPS industry coding.
#[pymodule]
mod soccernetpy {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use soccer_rs::{MODEL_CONFIG, ModelType, SoccerBuilder, SoccerPipeline, get_crosswalk};

    const SOCCERNET_OUTPUT_SYSTEM: &str = "soc2010";
    const CLIPS_OUTPUT_SYSTEM: &str = "naics2022";

    /// One ranked classification candidate.
    ///
    /// Attributes:
    ///     code (str): SOC 2010 or NAICS 2022 classification code.
    ///     title (str): Official title associated with ``code``.
    ///     score (float): Model score for this candidate. Larger values rank
    ///         ahead of smaller values within the same input row.
    #[pyclass(frozen, get_all)]
    #[derive(Debug)]
    struct SoccerResult {
        code: String,
        title: String,
        score: f32,
    }

    #[pymethods]
    impl SoccerResult {
        fn __repr__(&self) -> String {
            format!(
                "SoccerResult(code={:?}, title={:?}, score={:.4})",
                self.code, self.title, self.score
            )
        }
    }

    #[derive(FromPyObject)]
    enum PriorCodes {
        One(String),
        Many(Vec<String>),
    }

    /// Classify job descriptions into SOC 2010 occupations with SOCcerNET.
    ///
    /// Args:
    ///     job_titles (list[str]): Job titles to classify.
    ///     job_tasks (list[str]): Task descriptions corresponding positionally
    ///         to ``job_titles``.
    ///     soc1980 (list[str | list[str] | None] | None): Optional SOC
    ///         1980 prior codes, one entry per job.
    ///     isco1988 (list[str | list[str] | None] | None): Optional ISCO
    ///         1988 prior codes, one entry per job.
    ///     noc2011 (list[str | list[str] | None] | None): Optional NOC
    ///         2011 prior codes, one entry per job.
    ///     n (int): Maximum number of ranked candidates returned per job.
    ///         Defaults to 10.
    ///
    /// Returns:
    ///     list[list[SoccerResult]]: One candidate list per input job, in the
    ///     same order as ``job_titles``. Each inner list is ordered from the
    ///     highest-ranked candidate to the lowest-ranked candidate.
    ///
    /// Raises:
    ///     ValueError: If corresponding input lists have different lengths.
    ///     OverflowError: If ``n`` is negative or too large for ``usize``.
    ///     RuntimeError: If model setup, inference, or a crosswalk fails.
    ///
    /// Each prior entry may be a single code, multiple codes, or ``None``. For
    /// example: ``soc1980=["261", ["211", "212"], None]``.
    #[pyfunction]
    #[pyo3(signature = (
        job_titles,
        job_tasks,
        soc1980=None,
        isco1988=None,
        noc2011=None,
        n=10
    ))]
    fn soccernet(
        job_titles: Vec<String>,
        job_tasks: Vec<String>,
        soc1980: Option<Vec<Option<PriorCodes>>>,
        isco1988: Option<Vec<Option<PriorCodes>>>,
        noc2011: Option<Vec<Option<PriorCodes>>>,
        n: usize,
    ) -> PyResult<Vec<Vec<SoccerResult>>> {
        let number_of_jobs = job_titles.len();
        validate_length("job_tasks", job_tasks.len(), "job_titles", number_of_jobs)?;
        validate_optional_length("soc1980", &soc1980, "job_titles", number_of_jobs)?;
        validate_optional_length("isco1988", &isco1988, "job_titles", number_of_jobs)?;
        validate_optional_length("noc2011", &noc2011, "job_titles", number_of_jobs)?;

        if number_of_jobs == 0 {
            return Ok(Vec::new());
        }

        let mut prior = vec![Vec::new(); number_of_jobs];
        add_prior_codes(&soc1980, "soc1980", SOCCERNET_OUTPUT_SYSTEM, &mut prior)?;
        add_prior_codes(&isco1988, "isco1988", SOCCERNET_OUTPUT_SYSTEM, &mut prior)?;
        add_prior_codes(&noc2011, "noc2011", SOCCERNET_OUTPUT_SYSTEM, &mut prior)?;
        let prior: Vec<Box<[u16]>> = prior.into_iter().map(Vec::into_boxed_slice).collect();

        let config = MODEL_CONFIG
            .get_default_version(&ModelType::SOCcerNET)
            .ok_or_else(|| PyRuntimeError::new_err("SOCcerNET is not configured"))?;
        let output_system = config.output_system();
        let mut pipeline = SoccerPipeline::build(config).map_err(soccer_error)?;

        let ids: Vec<String> = (1..=number_of_jobs)
            .map(|index| format!("id-{index}"))
            .collect();
        let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let title_refs: Vec<&str> = job_titles.iter().map(String::as_str).collect();
        let task_refs: Vec<&str> = job_tasks.iter().map(String::as_str).collect();

        let coded_jobs = pipeline
            .predict_from_columns(&id_refs, &title_refs, Some(&task_refs), &prior)
            .map_err(soccer_error)?;

        coded_jobs
            .into_iter()
            .map(|job| {
                job.scored_code_index
                    .into_iter()
                    .take(n)
                    .map(|scored| {
                        let (code, title) = output_system
                            .get_code_title(scored.0 as u32)
                            .ok_or_else(|| {
                                PyRuntimeError::new_err(format!(
                                    "SOCcerNET returned unknown SOC 2010 index {}",
                                    scored.0
                                ))
                            })?;
                        Ok(SoccerResult {
                            code: code.to_owned(),
                            title: title.to_owned(),
                            score: scored.1,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Classify product and service descriptions into NAICS 2022 with CLIPS.
    ///
    /// Args:
    ///     products_services (list[str]): Product or service descriptions to
    ///         classify.
    ///     sic1987 (list[str | list[str] | None] | None): Optional SIC
    ///         1987 prior codes, one entry per description. Each entry may be a
    ///         single code, multiple codes, or ``None``.
    ///     n (int): Maximum number of ranked candidates returned per
    ///         description. Defaults to 10.
    ///
    /// Returns:
    ///     list[list[SoccerResult]]: One candidate list per input description,
    ///     in input order. Candidates are ordered from highest-ranked to
    ///     lowest-ranked.
    ///
    /// Raises:
    ///     ValueError: If ``sic1987`` and ``products_services`` have
    ///         different lengths.
    ///     OverflowError: If ``n`` is negative or too large for ``usize``.
    ///     RuntimeError: If model setup, inference, or the crosswalk fails.
    #[pyfunction]
    #[pyo3(signature = (products_services, sic1987=None, n=10))]
    fn clips(
        products_services: Vec<String>,
        sic1987: Option<Vec<Option<PriorCodes>>>,
        n: usize,
    ) -> PyResult<Vec<Vec<SoccerResult>>> {
        let number_of_jobs = products_services.len();
        validate_optional_length("sic1987", &sic1987, "products_services", number_of_jobs)?;

        if number_of_jobs == 0 {
            return Ok(Vec::new());
        }

        let mut prior = vec![Vec::new(); number_of_jobs];
        add_prior_codes(&sic1987, "sic1987", CLIPS_OUTPUT_SYSTEM, &mut prior)?;
        let prior: Vec<Box<[u16]>> = prior.into_iter().map(Vec::into_boxed_slice).collect();

        let config = MODEL_CONFIG
            .get_default_version(&ModelType::CLIPS)
            .ok_or_else(|| PyRuntimeError::new_err("CLIPS is not configured"))?;
        let output_system = config.output_system();
        let mut pipeline = SoccerPipeline::build(config).map_err(soccer_error)?;

        let ids: Vec<String> = (1..=number_of_jobs)
            .map(|index| format!("id-{index}"))
            .collect();
        let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let product_service_refs: Vec<&str> =
            products_services.iter().map(String::as_str).collect();

        let coded_jobs = pipeline
            .predict_from_columns(&id_refs, &product_service_refs, None, &prior)
            .map_err(soccer_error)?;

        coded_jobs
            .into_iter()
            .map(|job| {
                job.scored_code_index
                    .into_iter()
                    .take(n)
                    .map(|scored| {
                        let (code, title) = output_system
                            .get_code_title(scored.0 as u32)
                            .ok_or_else(|| {
                                PyRuntimeError::new_err(format!(
                                    "CLIPS returned unknown NAICS 2022 index {}",
                                    scored.0
                                ))
                            })?;
                        Ok(SoccerResult {
                            code: code.to_owned(),
                            title: title.to_owned(),
                            score: scored.1,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    fn validate_optional_length<T>(
        name: &str,
        values: &Option<Vec<T>>,
        expected_name: &str,
        expected: usize,
    ) -> PyResult<()> {
        if let Some(values) = values {
            validate_length(name, values.len(), expected_name, expected)?;
        }
        Ok(())
    }

    fn validate_length(
        name: &str,
        actual: usize,
        expected_name: &str,
        expected: usize,
    ) -> PyResult<()> {
        if actual != expected {
            return Err(PyValueError::new_err(format!(
                "{name} has {actual} rows, but {expected_name} has {expected}"
            )));
        }
        Ok(())
    }

    fn add_prior_codes(
        codes: &Option<Vec<Option<PriorCodes>>>,
        source_system: &str,
        output_system: &str,
        prior: &mut [Vec<u16>],
    ) -> PyResult<()> {
        let Some(codes) = codes else {
            return Ok(());
        };

        let crosswalk = get_crosswalk(source_system, output_system).map_err(soccer_error)?;
        codes.iter().zip(prior.iter_mut()).for_each(|(row, out)| {
            let code_refs: Vec<&str> = match row {
                Some(PriorCodes::One(code)) => vec![code.as_str()],
                Some(PriorCodes::Many(codes)) => codes.iter().map(String::as_str).collect(),
                None => Vec::new(),
            };
            crosswalk.crosswalk_into(&code_refs, out);
        });
        Ok(())
    }

    fn soccer_error(error: soccer_rs::MyError) -> PyErr {
        PyRuntimeError::new_err(error.to_string())
    }
}
