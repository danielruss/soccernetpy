"""Fetch the pinned Microsoft ONNX Runtime native libraries for wheel builds."""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ORT_VERSION = "1.24.2"
ORT_SHA256 = "782564d3d68e87269ea1812cc94710ad06fe014faf093f01a6f89d0bd58719d3"
ORT_URL = (
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/"
    f"{ORT_VERSION}/microsoft.ml.onnxruntime.{ORT_VERSION}.nupkg"
)
RUNTIME_FILES = {
    "linux-x64": ("libonnxruntime.so", "libonnxruntime_providers_shared.so"),
    "linux-arm64": ("libonnxruntime.so", "libonnxruntime_providers_shared.so"),
    "win-x64": ("onnxruntime.dll", "onnxruntime_providers_shared.dll"),
    "win-arm64": ("onnxruntime.dll", "onnxruntime_providers_shared.dll"),
}


def fetch_onnxruntime(runtime, destination):
    """Verify the NuGet archive and copy only the requested runtime and notices."""
    files = RUNTIME_FILES[runtime]
    members = {f"runtimes/{runtime}/native/{name}": name for name in files}
    members.update({"LICENSE": "LICENSE", "ThirdPartyNotices.txt": "ThirdPartyNotices.txt"})
    destination = Path(destination)

    with tempfile.TemporaryDirectory(prefix="soccernetpy-onnxruntime-") as tmp:
        package = Path(tmp) / "onnxruntime.nupkg"
        print(f"Downloading ONNX Runtime {ORT_VERSION} for {runtime}", flush=True)
        urllib.request.urlretrieve(ORT_URL, package)
        digest = hashlib.sha256()
        with package.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != ORT_SHA256:
            raise ValueError("ONNX Runtime NuGet package checksum mismatch")

        with zipfile.ZipFile(package) as archive:
            # Check every requested member before writing any output files.
            for member in members:
                archive.getinfo(member)
            destination.mkdir(parents=True, exist_ok=True)
            for member, name in members.items():
                with archive.open(member) as source, (destination / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
                print(f"Bundled {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, choices=RUNTIME_FILES)
    parser.add_argument("--destination", type=Path, default=Path("python/soccernetpy"))
    args = parser.parse_args()
    fetch_onnxruntime(args.runtime, args.destination)


if __name__ == "__main__":
    main()
