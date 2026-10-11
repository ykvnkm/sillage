"""Check that a running sillage service actually works, end to end.

    python scripts/smoke_test.py http://localhost:8000
    python scripts/smoke_test.py https://<public address> --wait 900

Standard library only. CI runs it against the freshly built container on every pull
request; once the service has a public address, the same check runs against it.
Unit tests prove the code; this proves the *image*: system libraries present, model
downloaded and loadable, static files packaged. Those are exactly the failures unit
tests cannot see (journal E-014: the drawing module crashed at import on a missing
libXrender).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

VANILLIN = "COc1cc(C=O)ccc1O"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--wait", type=int, default=60, help="seconds to wait for /health")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    wait_for_health(base, args.wait)

    checks = [
        ("web page", lambda: "query-form" in get(f"{base}/")[1]),
        ("drawing", lambda: get(f"{base}/depict?smiles=CCO")[1].lstrip().startswith("<svg")),
        ("vanillin smells of vanilla", lambda: predicts_vanilla(base)),
        ("bromine is refused", lambda: refuses(base, "BrCCCCCC=O", "unseen_element")),
        ("model description", lambda: json.loads(get(f"{base}/model")[1])["descriptors"]),
    ]
    failed = 0
    for name, check in checks:
        try:
            ok = bool(check())
        except Exception as error:  # report every check, not just the first failure
            ok = False
            name = f"{name}: {error}"
        print(f"{'ok' if ok else 'FAILED':<7} {name}")
        failed += not ok
    return 1 if failed else 0


def wait_for_health(base: str, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            status, body = get(f"{base}/health")
            if status == 200:
                print(f"ok      health: {body.strip()}")
                return
        except OSError:
            pass
        if time.monotonic() > deadline:
            sys.exit(f"FAILED  {base}/health did not answer within {seconds}s")
        time.sleep(10)


def predicts_vanilla(base: str) -> bool:
    status, body = post(f"{base}/predict", {"smiles": VANILLIN})
    names = [d["descriptor"] for d in json.loads(body)["predicted"]]
    return status == 200 and names[:1] == ["vanilla"]


def refuses(base: str, smiles: str, reason: str) -> bool:
    status, body = post(f"{base}/predict", {"smiles": smiles})
    return status == 422 and json.loads(body)["reason"] == reason


def get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.read().decode("utf-8")


def post(url: str, payload: dict) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
