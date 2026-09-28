"""Download the model weights and verify them against `checkpoint/SHA256SUMS`.

The two large artifacts -- the masked-diffusion checkpoint and the XGBoost
regressor -- are not in git. They ship as assets on a GitHub Release, and this
script fetches them on demand.

They are not in git because git-lfs is the wrong mechanism for this
submission. The organizers validate by running a plain `git clone`; on a
machine without git-lfs installed, or after the repository's LFS bandwidth
quota is spent, an LFS-tracked file clones as a ~130-byte pointer and the
entry point fails while loading a model that looks present. A Release asset
downloads over plain HTTPS with no client-side tooling, and the checksum file
that IS committed makes tampering or truncation detectable.

`ampx.generate.main` calls `ensure_weights()` before loading anything, so a
clean clone works with no manual step. `scripts/fetch_weights.py` is a thin
command-line wrapper around this module -- the logic lives in the package
because `uv` installs only `src/ampx`, so an entry point cannot import from
`scripts/`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoint"
SUMS_FILE = CHECKPOINT_DIR / "SHA256SUMS"

#: Release tag carrying the assets. Override with AMPX_WEIGHTS_BASE_URL when
#: testing against a fork or a local file:// mirror.
DEFAULT_BASE_URL = os.environ.get(
    "AMPX_WEIGHTS_BASE_URL",
    "https://github.com/skadadi850/AMP_IT_UP/releases/download/weights-v1",
)

#: Only these are fetched. Everything else in SHA256SUMS is small enough to be
#: committed, and is verified in place rather than downloaded.
FETCHED = ("masked_diffusion_best.pt", "mic_regressor.json")

_CHUNK = 1 << 20


def read_expected_sums() -> dict[str, str]:
    """Parse `checkpoint/SHA256SUMS` into {filename: digest}."""
    if not SUMS_FILE.exists():
        raise SystemExit(f"missing {SUMS_FILE}; cannot verify anything")
    sums: dict[str, str] = {}
    for line in SUMS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition(" ")
        sums[name.strip().lstrip("*")] = digest
    return sums


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_like_lfs_pointer(path: Path) -> bool:
    """True if this is a git-lfs pointer rather than the real artifact.

    Worth naming explicitly: a pointer is a valid small text file, so without
    this check the failure surfaces much later as an unintelligible parse
    error from torch or xgboost.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(64).startswith(b"version https://git-lfs.github.com/spec/")
    except OSError:
        return False


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"could not download {url} (HTTP {exc.code}). If the release is "
            "not published yet, place the file at "
            f"{dest} by hand, or set AMPX_WEIGHTS_BASE_URL."
        ) from exc
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"could not reach {url}: {exc.reason}") from exc
    # Rename only after a complete download, so an interrupted run never
    # leaves a truncated file that looks like a finished one.
    tmp.replace(dest)


def ensure_weights(quiet: bool = False, base_url: str = DEFAULT_BASE_URL) -> None:
    """Make every artifact in SHA256SUMS present and correct.

    Idempotent and offline-safe: a file already present with the right digest
    is left alone and nothing is requested over the network.
    """
    expected = read_expected_sums()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def say(message: str) -> None:
        if not quiet:
            print(message)

    for name, digest in expected.items():
        dest = CHECKPOINT_DIR / name

        if dest.exists() and _looks_like_lfs_pointer(dest):
            say(f"{name}: git-lfs pointer, not the artifact; re-downloading")
            dest.unlink()

        if dest.exists():
            actual = sha256(dest)
            if actual == digest:
                say(f"{name}: ok")
                continue
            if name not in FETCHED:
                raise SystemExit(
                    f"{name} is committed to the repository but its sha256 "
                    f"does not match SHA256SUMS\n  expected {digest}\n"
                    f"  actual   {actual}"
                )
            say(f"{name}: checksum mismatch, re-downloading")
            dest.unlink()

        if name not in FETCHED:
            raise SystemExit(
                f"{name} is listed in SHA256SUMS but missing, and it is not a "
                "downloadable asset. It should have been committed."
            )

        url = f"{base_url}/{name}"
        say(f"{name}: downloading from {url}")
        _download(url, dest)

        actual = sha256(dest)
        if actual != digest:
            dest.unlink(missing_ok=True)
            raise SystemExit(
                f"{name} failed verification after download\n"
                f"  expected {digest}\n  actual   {actual}"
            )
        say(f"{name}: downloaded and verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    ensure_weights(quiet=args.quiet, base_url=args.base_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
