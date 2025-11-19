#!/usr/bin/env python3
"""
Parallel downloader for large Zenodo files (no external dependencies).

- Uses HTTP Range requests
- Splits file into N chunks and downloads them in parallel
- Merges chunks into final file
- Optional MD5 verification
"""

import hashlib
import math
import os
import sys
import threading
import time
import urllib.request

# --- MemBrain-seg Training data URLs ---------------------------------------

RECORD_ID = "15089686"
FILENAME = "MemBrain_seg_training_data.zip"
DOWNLOAD_URL = f"https://zenodo.org/records/{RECORD_ID}/files/{FILENAME}?download=1"

# Set to None if you don't care about checksum
MD5_EXPECTED = "4553547a60fd5afbda547f54d034ad28"

NUM_WORKERS = 8  # Increase if you have good bandwidth/CPU
CHUNK_SIZE = 1024 * 1024  # 1 MB per read

# ---------------------------------------------------------------------------


def get_filesize(url: str) -> int:
    """Ask server for Content-Length via HEAD."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        size = resp.getheader("Content-Length")
        if size is None:
            raise RuntimeError("Server did not provide Content-Length header.")
        return int(size)


def download_range(
    url: str,
    start: int,
    end: int,
    part_path: str,
    progress: dict,
    lock: threading.Lock,
    retries: int = 3,
):
    """Download byte range [start, end] to part_path."""
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as resp, open(part_path, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    with lock:
                        progress["downloaded"] += len(chunk)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            # simple exponential backoff
            time.sleep(2**attempt)


def print_progress(progress: dict, total_size: int, lock: threading.Lock):
    """Simple console progress bar."""
    while True:
        with lock:
            downloaded = progress["downloaded"]
            done = progress["done"]
        percent = downloaded / total_size * 100
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "#" * filled + "." * (bar_len - filled)
        sys.stdout.write(
            f"\r[{bar}] {percent:6.2f}%  "
            f"{downloaded / (1024**2):7.2f} / {total_size / (1024**2):.2f} MB"
        )
        sys.stdout.flush()
        if done:
            break
        time.sleep(0.5)
    print()  # newline


def merge_parts(part_paths, out_path):
    """Concatenate part files into final file."""
    with open(out_path, "wb") as outfile:
        for part in part_paths:
            with open(part, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    outfile.write(chunk)


def md5sum(path: str) -> str:
    """Calculate MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file_single_process(out_path, chunk_size=1024 * 1024):
    """Download file using a single process.

    Parameters
    ----------
    out_path : str
        Path to save the downloaded file.
    chunk_size : int
        Number of bytes to read per chunk.
    """
    print(f"Downloading to {out_path}")
    with urllib.request.urlopen(DOWNLOAD_URL) as response:
        total = response.length
        downloaded = 0

        with open(out_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

                downloaded += len(chunk)
                if total:
                    done = int(50 * downloaded / total)
                    bar = "[" + "#" * done + "." * (50 - done) + "]"
                    print(
                        f"\r{bar} "
                        f"{downloaded / (1024**2):.2f} / {total / (1024**2):.2f} MB",
                        end="",
                        flush=True,
                    )
    print("\nDownload finished.")


def download_file_parallel(out_path: str, num_workers: int = 2):
    """
    Download file in parallel using multiple threads.

    Parameters
    ----------
    out_path : str
        Path to save the downloaded file.
    num_workers : int
        Number of parallel download threads to use.
    """
    print("Querying file size from Zenodo…")
    total_size = get_filesize(DOWNLOAD_URL)
    print(f"Total size: {total_size / (1024**3):.2f} GB")

    part_size = math.ceil(total_size / num_workers)
    part_paths = [f"{out_path}.part{i}" for i in range(num_workers)]

    progress = {"downloaded": 0, "done": False}
    lock = threading.Lock()

    # Start progress thread
    prog_thread = threading.Thread(
        target=print_progress,
        args=(progress, total_size, lock),
        daemon=True,
    )
    prog_thread.start()

    # Start worker threads
    threads = []
    for i in range(num_workers):
        start = i * part_size
        end = min(total_size - 1, (i + 1) * part_size - 1)
        if start > end:
            break  # in case num_workers > needed
        t = threading.Thread(
            target=download_range,
            args=(DOWNLOAD_URL, start, end, part_paths[i], progress, lock),
        )
        threads.append(t)
        t.start()

    # Wait for workers
    for t in threads:
        t.join()

    # Flag progress as done
    with lock:
        progress["done"] = True
    prog_thread.join()

    print("Merging parts…")
    merge_parts(part_paths, out_path)

    # Cleanup part files
    for p in part_paths:
        if os.path.exists(p):
            os.remove(p)

    print("Download finished.")


def download_data(out_folder, num_parallel_processes=1):
    """Download the MemBrain-seg dataset from Zenodo.

    Parameters
    ----------
    out_folder : str
        Folder to store the downloaded MemBrain-seg dataset (14GB .zip file).
    num_parallel_processes : int
        Number of parallel processes to use for downloading the dataset.
        Since dataset is hosted on Zenodo with limited bandwidth,
        using multiple parallel processes can speed up the download.
        Pass an integer value greater than 0. For example, to use 4 parallel
        processes, pass "4". Default is "1" (single process).
    """
    out_path = os.path.join(out_folder, FILENAME)
    out_path = FILENAME

    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping download.")
    elif num_parallel_processes <= 1:
        print("Downloading using single process…")
        print(
            "Note: Download via Zenodo may be slow due to server limits. \
                When in doubt, better get a coffee."
        )
        download_file_single_process(out_path)
    else:
        print(f"Downloading using {num_parallel_processes} parallel processes…")
        print(
            "Note: Download via Zenodo may be slow due to server limits. \
                When in doubt, better get a coffee."
        )
        download_file_parallel(out_path, num_workers=num_parallel_processes)

    # Optional MD5 verification
    if MD5_EXPECTED:
        print("\nVerifying MD5 checksum…")
        md5_actual = md5sum(out_path)
        print(f"Expected: {MD5_EXPECTED}")
        print(f"Actual  : {md5_actual}")
        if md5_actual.lower() == MD5_EXPECTED.lower():
            print("✅ MD5 checksum matches. File OK.")
        else:
            print("❌ MD5 checksum does NOT match! File may be corrupted.")
