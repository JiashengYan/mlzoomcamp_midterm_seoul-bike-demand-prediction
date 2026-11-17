"""Read CSV data across platforms without enforcing a byte fingerprint."""
import importlib.util

import pytest

from seoul_bike.data import RAW_PATH, ROOT, load_data


spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts/download_data.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_source_line_endings(tmp_path, newline):
    """Accept both platform line endings without rewriting the source.

    Inputs:
        tmp_path: Temporary directory for a source copy.
        newline: LF or CRLF byte sequence.

    Outputs:
        None. Asserts validation, parsing, and preservation of file bytes.
    """
    payload = RAW_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline)
    path = tmp_path / "source.csv"
    path.write_bytes(payload)
    assert downloader.validate_csv(payload) == (8760, 14)
    assert load_data(path).shape == (8760, 13)
    assert path.read_bytes() == payload


def test_source_without_fingerprint_check(tmp_path):
    """Read an equivalent CSV with a different byte representation.

    Inputs:
        tmp_path: Temporary directory for the altered source.

    Outputs:
        None. Asserts download checks and runtime parsing accept different bytes.
    """
    payload = RAW_PATH.read_bytes().replace(b",", b',"', 1)
    payload = payload.replace(b',"Rented Bike Count,', b',"Rented Bike Count",', 1)
    path = tmp_path / "changed.csv"
    path.write_bytes(payload)
    assert downloader.validate_csv(payload) == (8760, 14)
    assert load_data(path).equals(load_data())
