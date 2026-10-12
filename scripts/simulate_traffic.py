"""Send realistic traffic to a running service, then shift it, so monitoring has a story.

    python scripts/simulate_traffic.py                      # 8 min normal, then 8 min drift
    python scripts/simulate_traffic.py --normal 2 --drift 2 --rate 8

Standard library only. The "normal" phase asks about molecules the service knows --
by name and by SMILES, taken from its own /map -- with a few typos. The "drift" phase
models a different audience arriving: pharmacology students pasting drugs (in scope,
but far from any odorant), halogenated solvents (an element the model never saw),
salts. The dashboard should show it: similarity falls below the dataset line,
out-of-scope refusals climb, confidence drops.

There is no real traffic yet (the service is not public, ADR-0015), so this is how
the monitoring is exercised. It is synthetic, and the dashboard screenshot in the
README says so.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request

# In scope (C, N, O, S; under 400 Da) but nothing like a perfumery material.
DRUGS = [
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CN1CCCC1c1cccnc1",  # nicotine
    "CCN(CC)CC(=O)Nc1c(C)cccc1C",  # lidocaine
    "CN(C)C(=N)N=C(N)N",  # metformin
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",  # procaine
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",  # atenolol
    "CC(C)NCC(O)COc1cccc2ccccc12",  # propranolol
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",  # naproxen
    "Nc1ccc(cc1)S(N)(=O)=O",  # sulfanilamide
    "CC(CS)C(=O)N1CCCC1C(=O)O",  # captopril
    "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O",  # phenobarbital
    "CNC(C)C(O)c1ccccc1",  # ephedrine
    "NC(Cc1c[nH]c2ccccc12)C(=O)O",  # tryptophan
    "Nc1ncnc2[nH]cnc12",  # adenine
    "OC(=O)CCC(=O)O",  # succinic acid
]
# Elements the model never saw, and things that are not one neutral molecule.
OUT_OF_SCOPE = [
    "Clc1ccccc1",  # chlorobenzene
    "ClC(Cl)Cl",  # chloroform
    "BrC(Br)Br",  # bromoform
    "Ic1ccccc1",  # iodobenzene
    "FC(F)(F)c1ccccc1",  # benzotrifluoride
    "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",  # fluoxetine
    "Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl",  # triclosan
    "[Na+].[Cl-]",
    "[K+].[O-]C(=O)c1ccccc1",  # potassium benzoate
]
TYPOS = ["ванилн", "COc1cc(C=O", "hello", "линалол", "CCCC((O"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--normal", type=float, default=8.0, help="minutes of normal traffic")
    parser.add_argument("--drift", type=float, default=8.0, help="minutes of drifted traffic")
    parser.add_argument("--rate", type=float, default=4.0, help="requests per second")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = args.url.rstrip("/")
    with urllib.request.urlopen(f"{base}/map", timeout=30) as response:
        points = json.loads(response.read())["points"]
    known = [(s, n) for s, n in zip(points["smiles"], points["name"], strict=True)]

    def normal() -> str:
        roll = rng.random()
        smiles, name = rng.choice(known)
        if roll < 0.05:
            return rng.choice(TYPOS)
        if roll < 0.55 and name:
            return name
        return smiles

    def drifted() -> str:
        roll = rng.random()
        if roll < 0.45:
            return rng.choice(DRUGS)
        if roll < 0.85:
            return rng.choice(OUT_OF_SCOPE)
        return normal()

    run(base, "normal", normal, args.normal, args.rate)
    run(base, "drift", drifted, args.drift, args.rate)
    return 0


def run(base: str, phase: str, next_query, minutes: float, rate: float) -> None:
    deadline = time.monotonic() + minutes * 60
    counts: dict[int, int] = {}
    last_report = time.monotonic()
    while time.monotonic() < deadline:
        status = post(f"{base}/predict", {"query": next_query()})
        counts[status] = counts.get(status, 0) + 1
        if time.monotonic() - last_report > 30:
            print(f"{phase:<7} {dict(sorted(counts.items()))}", flush=True)
            last_report = time.monotonic()
        time.sleep(1 / rate)
    print(f"{phase:<7} done {dict(sorted(counts.items()))}", flush=True)


def post(url: str, payload: dict) -> int:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except OSError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
