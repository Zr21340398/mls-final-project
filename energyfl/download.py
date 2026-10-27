"""Official LCL download with resumable transfer, safe extraction and provenance.

No pandas/torch import is required for this command. Hashes establish local
integrity, not a publisher signature (the source does not publish a SHA256 here).
"""

import contextlib
import os
import re
import shutil
import ssl
import stat
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from .common import digest, read_json, write_json

SOURCE_PAGE = "https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d"
LCL_URL = "https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/Partitioned%20LCL%20Data.zip"
BLOCK = 1024 * 1024


def verified_ssl_context():
    """Use native certificate-chain validation, including intermediate retrieval.

    Windows CryptoAPI can build chains that a bundled OpenSSL runtime cannot.
    Keep hostname verification and certificate verification enabled. An optional
    CA file extends trust for a user-managed institutional proxy; it is never an
    instruction to accept arbitrary server certificates.
    """
    try:
        import truststore
    except ImportError as exc:
        raise RuntimeError(
            "Missing TLS dependency. Install project requirements with the project Python."
        ) from exc
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    cafile = os.environ.get("ENERGYFL_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if cafile:
        context.load_verify_locations(cafile=cafile)
    return context


@contextlib.contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"Operation already locked: {path}. If a process was killed, confirm it stopped before removing this specific lock."
        )
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _validate_zip(path):
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a valid ZIP archive: {path}")
    # CRC verification occurs while each member is streamed during extraction;
    # avoid decompressing the entire 10+ GB archive twice just to check CRCs.
    with zipfile.ZipFile(path) as z:
        if not z.infolist():
            raise ValueError("Empty archive")


def _transfer(url, archive, timeout=60, retries=3, expected_sha256=None):
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    receipt = archive.with_suffix(".download.json")
    part = archive.with_suffix(".part")
    partial = archive.with_suffix(".partial.json")
    if archive.exists():
        if not receipt.exists():
            raise FileExistsError(
                f"Archive without receipt: {archive}; use a fresh output directory"
            )
        saved = read_json(receipt)
        if saved["url"] != url or digest(archive) != saved["sha256"]:
            raise ValueError("Cached archive failed provenance/integrity check")
        if expected_sha256 and saved["sha256"] != expected_sha256:
            raise ValueError("Expected SHA256 mismatch")
        print(f"Using verified cached archive: {archive}", flush=True)
        return saved
    tls_context = verified_ssl_context() if url.lower().startswith("https://") else None
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        prior = read_json(partial) if partial.exists() else {}
        # Without a matching URL and server validator, do not concatenate bytes
        # from possibly different archive versions. Restart safely instead.
        validator = prior.get("etag") or prior.get("last_modified")
        resumable = offset and prior.get("url") == url and validator
        headers = {
            "User-Agent": "EnergyFL/0.1 (academic public dataset downloader)",
            "Accept-Encoding": "identity",
        }
        if resumable:
            headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(
                request, timeout=timeout, context=tls_context
            ) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if not resumable or not match or int(match[1]) != offset:
                        raise ValueError("Server returned an unexpected Content-Range")
                    total = int(match[3])
                    mode = "ab"
                    current = response.headers.get("ETag") or response.headers.get(
                        "Last-Modified"
                    )
                    if current != validator:
                        raise ValueError(
                            "Archive validator changed during resume; choose a fresh download directory"
                        )
                elif status == 200:
                    offset = 0
                    mode = "wb"
                    length = response.headers.get("Content-Length")
                    total = int(length) if length else None
                else:
                    raise ValueError(f"Unexpected HTTP status: {status}")
                if (
                    total
                    and shutil.disk_usage(archive.parent).free
                    < max(0, total - offset) + 128 * 1024**2
                ):
                    raise OSError("Insufficient disk space for archive")
                info = {
                    "url": url,
                    "resolved_url": response.url,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "total_bytes": total,
                }
                write_json(partial, info)
                written = offset
                last = time.monotonic()
                with part.open(mode) as f:
                    while True:
                        block = response.read(BLOCK)
                        if not block:
                            break
                        f.write(block)
                        written += len(block)
                        if time.monotonic() - last >= 5:
                            print(
                                f"Downloaded {written / 1024**2:.1f} MiB"
                                + (f" / {total / 1024**2:.1f} MiB" if total else ""),
                                flush=True,
                            )
                            last = time.monotonic()
                    f.flush()
                    os.fsync(f.fileno())
                if total is not None and written != total:
                    raise OSError(f"Incomplete transfer: {written}/{total} bytes")
            sha = digest(part)
            if expected_sha256 and sha != expected_sha256:
                raise ValueError("Downloaded archive SHA256 mismatch")
            _validate_zip(part)
            info.update(
                sha256=sha,
                bytes=written,
                downloaded_utc=datetime.now(timezone.utc).isoformat(),
                source_page=SOURCE_PAGE,
                attribution="UK Power Networks, Low Carbon London; London Datastore; Creative Commons Attribution",
            )
            # Write receipt before rename: an interrupted transfer cannot be
            # mistaken for an accepted, complete archive on the next invocation.
            write_json(receipt, info)
            os.replace(part, archive)
            partial.unlink(missing_ok=True)
            return info
        except urllib.error.HTTPError as exc:
            # A previously complete .part may receive 416 on an interrupted final
            # commit. It will be re-fetched, never accepted without verification.
            if exc.code == 416:
                partial.unlink(missing_ok=True)
            elif exc.code < 500 and exc.code != 429:
                raise
            if attempt + 1 == retries:
                raise
        except (OSError, urllib.error.URLError) as exc:
            if isinstance(getattr(exc, "reason", exc), ssl.SSLCertVerificationError):
                # Certificate configuration errors are not transient. Repeating
                # the same request three times only hides the useful diagnosis.
                raise RuntimeError(
                    "HTTPS certificate verification failed using the system trust store. "
                    "Check the system clock and any institutional proxy CA. If your IT "
                    "provides a trusted PEM CA bundle, set ENERGYFL_CA_BUNDLE to that file. "
                    "Certificate verification has NOT been disabled."
                ) from exc
            if attempt + 1 == retries:
                raise
            print(
                f"Transfer interrupted ({exc}); retry {attempt + 2}/{retries}",
                flush=True,
            )
        time.sleep(min(2**attempt, 4))
    raise RuntimeError("Download retries exhausted")


def _members(z):
    result = []
    names = set()
    total = 0
    for member in z.infolist():
        if member.is_dir():
            continue
        name = member.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or any(":" in x for x in path.parts)
        ):
            raise ValueError("Unsafe archive member path")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError("ZIP symbolic links are not supported")
        if "__MACOSX" in path.parts or path.suffix.lower() != ".csv":
            continue
        basename = path.name
        if basename.lower() in names:
            raise ValueError("Duplicate CSV basename in archive")
        names.add(basename.lower())
        total += member.file_size
        result.append((member, basename))
    if not result:
        raise ValueError("No CSV data found in archive")
    if len(result) > 1000 or total > 25 * 1024**3:
        raise ValueError(
            "Unexpected archive dimensions; inspect source before proceeding"
        )
    return result, total


def extract_archive(archive, out, archive_sha256=None):
    """Flatten CSV names into an owned staging directory; never use extractall."""
    archive = Path(archive)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sha = archive_sha256 or digest(archive)
    if out.exists():
        manifest_path = out / "extraction.json"
        if not manifest_path.exists():
            raise FileExistsError(
                "Extraction target is not a verified cache; choose another --out directory"
            )
        manifest = read_json(manifest_path)
        if manifest["archive_sha256"] != sha:
            raise ValueError("Archive changed; extraction cache is stale")
        for f in manifest["files"]:
            path = (out / f["file"]).resolve()
            if (
                not path.is_relative_to(out.resolve())
                or not path.is_file()
                or digest(path) != f["sha256"]
            ):
                raise ValueError("Extracted CSV failed integrity check")
        return manifest
    staging = out.with_name(out.name + ".incomplete")
    staging.mkdir(exist_ok=True)
    progress_path = staging / "extraction.partial.json"
    if any(staging.iterdir()) and not progress_path.exists():
        # An interruption after the final manifest was written can be completed.
        if (staging / "extraction.json").exists():
            manifest = read_json(staging / "extraction.json")
            if manifest["archive_sha256"] != sha:
                raise ValueError("Incomplete extraction provenance mismatch")
            for f in manifest["files"]:
                path = (staging / f["file"]).resolve()
                if (
                    not path.is_relative_to(staging.resolve())
                    or not path.is_file()
                    or digest(path) != f["sha256"]
                ):
                    raise ValueError("Incomplete extraction integrity failure")
            os.replace(staging, out)
            return manifest
        raise FileExistsError(
            "Unrecognized extraction staging directory; choose a new output directory"
        )
    progress = (
        read_json(progress_path)
        if progress_path.exists()
        else {"archive_sha256": sha, "files": []}
    )
    if progress["archive_sha256"] != sha:
        raise ValueError(
            "Archive differs from incomplete extraction; use a new output directory"
        )
    write_json(progress_path, progress)
    done = {f["file"]: f for f in progress["files"]}
    with zipfile.ZipFile(archive) as z:
        members, total = _members(z)
        cached = sum(f["bytes"] for f in done.values())
        if shutil.disk_usage(staging).free < max(0, total - cached) + 512 * 1024**2:
            raise OSError(
                f"Insufficient disk space; extraction needs up to {total / 1024**3:.2f} GiB"
            )
        for i, (member, name) in enumerate(members, 1):
            target = staging / name
            if not target.resolve().is_relative_to(staging.resolve()):
                raise ValueError("Unsafe extraction target")
            if (
                name in done
                and target.is_file()
                and digest(target) == done[name]["sha256"]
            ):
                continue
            tmp = target.with_suffix(target.suffix + ".part")
            # ZipExtFile checks CRC at EOF. An interrupted write is never marked
            # complete and can be safely replaced on retry.
            with z.open(member) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, BLOCK)
            if tmp.stat().st_size != member.file_size:
                raise ValueError("Uncompressed length mismatch")
            entry = {
                "file": name,
                "archive_member": member.filename,
                "bytes": member.file_size,
                "sha256": digest(tmp),
            }
            os.replace(tmp, target)
            done[name] = entry
            progress["files"] = list(done.values())
            write_json(progress_path, progress)
            print(f"Extracted CSV {i}/{len(members)}: {name}", flush=True)
    manifest = {
        "archive_sha256": sha,
        "uncompressed_bytes": total,
        "files": list(done.values()),
        "source_page": SOURCE_PAGE,
    }
    write_json(staging / "extraction.json", manifest)
    progress_path.unlink(missing_ok=True)
    os.replace(staging, out)
    return manifest


def download_lcl(out="data/raw/lcl", url=LCL_URL, extract=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / "DOWNLOAD.lock"):
        archive = out / "lcl-168.zip"
        receipt = _transfer(url, archive)
        if extract:
            manifest = extract_archive(archive, out / "csv", receipt["sha256"])
            return {
                "archive": str(archive),
                "csv_dir": str(out / "csv"),
                "csv_files": len(manifest["files"]),
                "sha256": receipt["sha256"],
            }
        return {"archive": str(archive), "sha256": receipt["sha256"]}
