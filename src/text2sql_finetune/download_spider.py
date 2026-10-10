from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"


def _find_spider_root(path: Path) -> Path:
    if (path / "train_spider.json").exists() and (path / "tables.json").exists():
        return path
    for child in path.rglob("train_spider.json"):
        candidate = child.parent
        if (candidate / "tables.json").exists():
            return candidate
    raise FileNotFoundError("Cannot find train_spider.json and tables.json after extraction.")


def download_and_extract(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "spider.zip"
    print(f"Downloading Spider from {url}")
    urlretrieve(url, zip_path)
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    spider_root = _find_spider_root(out_dir)
    target = out_dir / "spider"
    if spider_root.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(spider_root), str(target))
    print(f"Spider dataset is ready at {target}")


def download_gdrive_and_extract(file_id: str, out_dir: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required for Google Drive download. Run pip install gdown.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "spider_data.zip"
    print(f"Downloading Spider Google Drive file id={file_id}")
    gdown.download(id=file_id, output=str(zip_path), quiet=False)
    if not zip_path.exists():
        raise RuntimeError("gdown did not create the zip file.")
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    spider_root = _find_spider_root(out_dir)
    target = out_dir / "spider"
    if spider_root.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(spider_root), str(target))
    print(f"Spider dataset is ready at {target}")


def copy_from_zip(zip_file: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(out_dir)
    spider_root = _find_spider_root(out_dir)
    target = out_dir / "spider"
    if spider_root.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(spider_root), str(target))
    print(f"Spider dataset is ready at {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="Direct Spider zip URL.")
    parser.add_argument("--gdrive-id", default=DEFAULT_GDRIVE_ID, help="Spider official Google Drive file id.")
    parser.add_argument("--zip-file", default=None, help="Local Spider zip file.")
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    try:
        if args.zip_file:
            copy_from_zip(Path(args.zip_file), out_dir)
        elif args.url:
            download_and_extract(args.url, out_dir)
        else:
            download_gdrive_and_extract(args.gdrive_id, out_dir)
    except Exception as exc:
        print(f"Download/extract failed: {exc}", file=sys.stderr)
        print(
            "Please manually download Spider and extract it to data/raw/spider. "
            "The directory must contain train_spider.json, dev.json, tables.json and database/.",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
