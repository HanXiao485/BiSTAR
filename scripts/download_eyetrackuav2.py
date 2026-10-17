#!/usr/bin/env python3
"""Download the official EyeTrackUAV2 release with resume and checksums."""

import argparse
import csv
import ftplib
import hashlib
import json
import os
import posixpath
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_HOST = "ftp.ivc.polytech.univ-nantes.fr"
DEFAULT_REMOTE_ROOT = "/EyeTrackUAV2"
DEFAULT_OUTPUT = Path("datasets/raw/EyeTrackUAV2")


def connect(host, timeout):
    ftp = ftplib.FTP(host, timeout=timeout)
    ftp.login()
    ftp.set_pasv(True)
    return ftp


def list_files(ftp, remote_root, reconnect, retries=5):
    files = []

    def walk(remote_dir):
        nonlocal ftp
        for attempt in range(1, retries + 1):
            try:
                entries = list(ftp.mlsd(remote_dir, facts=["type", "size", "modify"]))
                break
            except (ftplib.all_errors, OSError, socket.timeout) as error:
                if attempt == retries:
                    raise RuntimeError("cannot list {} after {} attempts".format(remote_dir, retries)) from error
                try:
                    ftp.close()
                except OSError:
                    pass
                ftp = reconnect()
        for name, facts in sorted(entries):
            if name in (".", ".."):
                continue
            remote_path = posixpath.join(remote_dir, name)
            kind = facts.get("type", "")
            if kind == "dir":
                walk(remote_path)
            elif kind == "file":
                files.append({
                    "remote_path": remote_path,
                    "relative_path": posixpath.relpath(remote_path, remote_root),
                    "size_bytes": int(facts.get("size", 0)),
                    "modified_utc": facts.get("modify", ""),
                })

    walk(remote_root.rstrip("/"))
    return files, ftp


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def download_file(ftp, item, output_root, retries=4):
    destination = output_root / item["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item["size_bytes"])
    if destination.is_file() and destination.stat().st_size == expected:
        return "existing", destination
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            partial.unlink()
            offset = 0
        try:
            with partial.open("ab" if offset else "wb") as handle:
                ftp.retrbinary("RETR " + item["remote_path"], handle.write, blocksize=1024 * 1024,
                               rest=offset if offset else None)
            if partial.stat().st_size != expected:
                raise IOError("downloaded size {} != expected {}".format(partial.stat().st_size, expected))
            os.replace(str(partial), str(destination))
            return "downloaded", destination
        except (ftplib.all_errors, OSError, socket.timeout) as error:
            if attempt == retries:
                raise RuntimeError("failed after {} attempts: {}".format(retries, item["remote_path"])) from error
            try:
                ftp.voidcmd("NOOP")
            except ftplib.all_errors:
                raise ConnectionError("FTP connection was lost") from error
    raise AssertionError("unreachable")


def write_inventory(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["remote_path", "relative_path", "size_bytes", "modified_utc"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--no-checksums", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ftp = connect(args.host, args.timeout)
    try:
        items, ftp = list_files(
            ftp, args.remote_root, lambda: connect(args.host, args.timeout))
        write_inventory(items, output / "download_inventory.csv")
        total = sum(item["size_bytes"] for item in items)
        print("Remote inventory: {} files, {} bytes ({:.3f} GiB)".format(
            len(items), total, total / float(1024 ** 3)), flush=True)
        if args.inventory_only:
            return 0
        completed = []
        for index, item in enumerate(items, 1):
            try:
                state, local_path = download_file(ftp, item, output)
            except ConnectionError:
                try:
                    ftp.quit()
                except ftplib.all_errors:
                    pass
                ftp = connect(args.host, args.timeout)
                state, local_path = download_file(ftp, item, output)
            record = dict(item)
            record.update({"local_path": str(local_path), "status": state})
            completed.append(record)
            print("[{}/{}] {} {}".format(index, len(items), state, item["relative_path"]), flush=True)

        if not args.no_checksums:
            for index, record in enumerate(completed, 1):
                record["sha256"] = sha256_file(Path(record["local_path"]))
                if index % 100 == 0 or index == len(completed):
                    print("Checksums: {}/{}".format(index, len(completed)), flush=True)
        metadata = {
            "dataset": "EyeTrackUAV2",
            "source_host": args.host,
            "remote_root": args.remote_root,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "file_count": len(completed),
            "total_bytes": total,
            "complete": len(completed) == len(items),
        }
        (output / "download_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fields = ["remote_path", "relative_path", "size_bytes", "modified_utc",
                  "local_path", "status", "sha256"]
        with (output / "download_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(completed)
        print(json.dumps(metadata, indent=2), flush=True)
        return 0
    finally:
        try:
            ftp.quit()
        except ftplib.all_errors:
            ftp.close()


if __name__ == "__main__":
    sys.exit(main())
