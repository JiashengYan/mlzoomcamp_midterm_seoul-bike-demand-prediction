"""Call a running local service using only Python's standard library."""
import argparse
import json
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--issued-at", default="2018-11-16T17:00:00+09:00")
args = parser.parse_args()
request = Request(args.url.rstrip("/") + "/predict",
                  data=json.dumps({"issued_at": args.issued_at}).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
with urlopen(request, timeout=30) as response:
    print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
