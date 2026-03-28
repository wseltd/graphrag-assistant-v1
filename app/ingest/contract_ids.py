"""Contract ID normalisation utilities.

Pure functions — no side effects, no I/O.
"""

import hashlib
import re
from pathlib import PurePosixPath, PureWindowsPath


def normalise_contract_id(filename: str) -> str:
    """Return a normalised contract ID derived from *filename*.

    Steps:
    1. Strip path components (keep only the final segment).
    2. Lowercase the result.
    3. Replace every run of non-alphanumeric characters with a single ``_``.
    4. Strip leading and trailing underscores.

    Examples::

        normalise_contract_id("contracts/ACME_Corp-2024.md") == "acme_corp_2024_md"
        normalise_contract_id("/abs/path/order_001.txt")      == "order_001_txt"
        normalise_contract_id("simple")                       == "simple"
    """
    # Strip path — handle both POSIX and Windows separators.
    # PurePosixPath covers the common case; fall back to name after splitting on backslash.
    basename = PurePosixPath(filename).name
    # Handle Windows-style paths that PurePosixPath treats as one segment.
    if "\\" in basename:
        basename = PureWindowsPath(basename).name

    lowered = basename.lower()
    underscored = re.sub(r"[^a-z0-9]+", "_", lowered)
    return underscored.strip("_")


def make_chunk_id(contract_id: str, chunk_index: int) -> str:
    """Return a deterministic SHA-256 hex digest for a chunk.

    The digest is computed over the UTF-8 encoding of
    ``f'{contract_id}:{chunk_index}'``.
    """
    payload = f"{contract_id}:{chunk_index}".encode()
    return hashlib.sha256(payload).hexdigest()
