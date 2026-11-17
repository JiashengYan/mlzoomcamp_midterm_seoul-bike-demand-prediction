"""Download the pinned original Seoul Bike CSV using only the standard library."""

import csv
import io
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile


SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/560/"
    "seoul+bike+sharing+demand.zip"
)
CSV_FILENAME = "SeoulBikeData.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "raw" / CSV_FILENAME
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024


def validate_csv(payload: bytes) -> tuple[int, int]:
    """Check that CSV bytes have the expected dimensions.

    Inputs:
        payload: Original CSV bytes decoded as latin-1 only for structural checks.

    Outputs:
        A (row_count, column_count) tuple equal to (8760, 14). Raises ValueError for
        invalid row/column counts; writes no files.
    """
    rows = list(csv.reader(io.StringIO(payload.decode("latin-1"))))
    if not rows or len(rows[0]) != 14 or len(rows) - 1 != 8760:
        raise ValueError("Unexpected CSV dimensions; expected 8760 rows and 14 columns.")
    if any(len(row) != 14 for row in rows[1:]):
        raise ValueError("CSV contains a row with an unexpected number of columns.")
    return len(rows) - 1, len(rows[0])


def main() -> int:
    """Download and verify the source CSV, or verify an existing local copy.

    Inputs:
        No function arguments. Uses SOURCE_URL, DESTINATION, and the
            archive-size limit defined in this module.

    Outputs:
        Integer exit status: 0 on success, 1 for a handled failure. Prints the result;
        creates a verified CSV only when absent and never overwrites an existing file.
    """
    try:
        if DESTINATION.exists():
            row_count, column_count = validate_csv(DESTINATION.read_bytes())
            action = "Verified existing file; download skipped"
        else:
            print(f"Downloading: {SOURCE_URL}", flush=True)
            with urlopen(SOURCE_URL, timeout=60) as response:
                archive_bytes = response.read(MAX_ARCHIVE_BYTES + 1)
            if len(archive_bytes) > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds the expected download size limit.")
            with ZipFile(io.BytesIO(archive_bytes)) as archive:
                entry = archive.getinfo(CSV_FILENAME)
                if entry.file_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("CSV exceeds the expected file size limit.")
                payload = archive.read(entry)
            row_count, column_count = validate_csv(payload)
            DESTINATION.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents overwriting an existing local file.
            with DESTINATION.open("xb") as output:
                output.write(payload)
            action = "Downloaded and verified"
        print(action)
        print(f"File: {DESTINATION}")
        print(f"Shape: {row_count} rows x {column_count} columns")
        return 0
    except (OSError, URLError, ValueError, BadZipFile, KeyError) as error:
        print(f"Data setup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
