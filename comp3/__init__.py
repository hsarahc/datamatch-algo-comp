# ENABLE_CHECK50_ASSERT = 1
import check50
import json
from pathlib import Path

def pair_key(a: str, b: str) -> str:
    a, b = sorted([a, b])
    return f"{a}-{b}"

@check50.check()
def exists():
    """main exists"""
    check50.exists("main.cc")

@check50.check(exists, timeout=30)
def compiles():
    """main compiles"""
    check50.run("make main") 

@check50.check(compiles)
def binary_present():
    """main binary was produced"""
    # Help debug local runs that report exit code 127 by ensuring the
    # binary exists and is listable in the sandbox.
    check50.run("ls -l main")

@check50.check(compiles)
def runs():
    """program runs successfully"""
    check50.run("./main data/FIREBASE_EXPORT.json data/sims/").exit(0)

@check50.check(runs)
def u1_u2_cannot_date_if_same_dorm_and_noDormMatch():
    """u1 and u2 cannot date when same dorm and noDormMatch=true"""
    p = Path("results/matches_Harvard.json")
    assert p.exists(), "results/matches_Harvard.json not found"

    matches = json.loads(p.read_text())
    key = pair_key("u1", "u2")

    if key not in matches:
        # Ideal outcome: rule blocked the match entirely.
        return

    # If present anyway, ensure canDate is false.
    entry = matches[key]
    can_date = entry.get("canDate")
    assert can_date == False, f"{key} present with canDate=true (should not be allowed)"
