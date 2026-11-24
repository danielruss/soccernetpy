import hashlib
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import fetch_onnxruntime as fetcher


class FetchOnnxRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / "runtime.nupkg"
        self.destination = self.root / "package"

    def make_archive(self, omit=None):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for runtime, files in fetcher.RUNTIME_FILES.items():
                for name in files:
                    member = f"runtimes/{runtime}/native/{name}"
                    if member != omit:
                        archive.writestr(member, f"{runtime}/{name}")
            archive.writestr("LICENSE", "license")
            archive.writestr("ThirdPartyNotices.txt", "notices")
        return hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def download(self, url, destination):
        self.assertEqual(url, fetcher.ORT_URL)
        shutil.copyfile(self.archive, destination)

    def test_extracts_only_selected_architecture_and_notices(self):
        digest = self.make_archive()
        for runtime, files in fetcher.RUNTIME_FILES.items():
            with self.subTest(runtime=runtime):
                destination = self.destination / runtime
                with patch.object(fetcher.urllib.request, "urlretrieve", self.download), patch.object(
                    fetcher, "ORT_SHA256", digest
                ):
                    fetcher.fetch_onnxruntime(runtime, destination)
                self.assertEqual(
                    {p.name for p in destination.iterdir()},
                    set(files) | {"LICENSE", "ThirdPartyNotices.txt"},
                )
                for name in files:
                    self.assertEqual((destination / name).read_text(), f"{runtime}/{name}")

    def test_checksum_mismatch_writes_no_runtime_files(self):
        self.make_archive()
        with patch.object(fetcher.urllib.request, "urlretrieve", self.download), patch.object(
            fetcher, "ORT_SHA256", "invalid"
        ):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fetcher.fetch_onnxruntime("win-x64", self.destination)
        self.assertFalse(self.destination.exists())

    def test_missing_member_writes_no_runtime_files(self):
        digest = self.make_archive(omit="runtimes/win-arm64/native/onnxruntime.dll")
        with patch.object(fetcher.urllib.request, "urlretrieve", self.download), patch.object(
            fetcher, "ORT_SHA256", digest
        ):
            with self.assertRaises(KeyError):
                fetcher.fetch_onnxruntime("win-arm64", self.destination)
        self.assertFalse(self.destination.exists())

    def test_download_failure_writes_no_runtime_files(self):
        with patch.object(fetcher.urllib.request, "urlretrieve", side_effect=OSError("offline")):
            with self.assertRaisesRegex(OSError, "offline"):
                fetcher.fetch_onnxruntime("linux-x64", self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
