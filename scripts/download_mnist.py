"""Download standard MNIST IDX files (same 60k/10k split as ylecun/mnist)."""

import gzip
import pathlib
import shutil
import urllib.request

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def main():
    dest = pathlib.Path("data")
    dest.mkdir(exist_ok=True)
    for name in FILES:
        path = dest / name
        if path.exists():
            print(f"exists: {path}")
            continue
        print(f"downloading: {name}.gz", flush=True)
        with urllib.request.urlopen(BASE + name + ".gz", timeout=60) as response:
            with gzip.GzipFile(fileobj=response) as compressed:
                with path.with_suffix(".part").open("wb") as output:
                    shutil.copyfileobj(compressed, output)
        path.with_suffix(".part").replace(path)


if __name__ == "__main__":
    main()
