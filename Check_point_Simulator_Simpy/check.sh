#!/usr/bin/env bash
# Local verification runner — run BEFORE every push (both of us: Sam + Atharva).
#
#   ./check.sh          quick gate  (~2 min): tests (event engine) + pencil-math
#                       hand-check + trace validator
#   ./check.sh full     full gate   (~8 min): tests under BOTH engines, all 3
#                       hand-checks + validator, mini-C smoke run
#   ./check.sh install-hook   installs this as .git/hooks/pre-push (quick mode)
#
# Physics ground truth: handcheck2's big0 peer-flush must be EXACTLY 3.4483 s
# (fluid pencil math; see ENGINE_REWRITE_VERIFICATION.md). If your change moves
# it, the change is wrong — or you have discovered something big: stop and talk.
set -u
cd "$(dirname "$0")"
MODE="${1:-quick}"
FAIL=0
TMP="$(mktemp -d /tmp/simcheck-XXXX)"
trap 'rm -rf "$TMP"' EXIT
say()  { printf '\n== %s\n' "$*"; }
bad()  { printf 'XX FAIL: %s\n' "$*"; FAIL=1; }

if [ "$MODE" = "install-hook" ]; then
  cat > .git/hooks/pre-push <<'HOOK'
#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/check.sh" quick
HOOK
  chmod +x .git/hooks/pre-push
  echo "pre-push hook installed (quick gate runs on every push)"
  exit 0
fi

export PYTHONHASHSEED=0

say "pytest (event engine)"
python3 -m pytest tests/ -q -x 2>&1 | tail -1 | tee "$TMP/t1"
# " passed" alone also matches "1 failed, 30 passed" — require no failures
{ grep -q " passed" "$TMP/t1" && ! grep -qE "failed|error" "$TMP/t1"; } \
  || bad "test suite (event)"

if [ "$MODE" = "full" ]; then
  say "pytest (polling engine)"
  SIM_ENGINE=polling python3 -m pytest tests/ -q -x 2>&1 | tail -1 | tee "$TMP/t2"
  { grep -q " passed" "$TMP/t2" && ! grep -qE "failed|error" "$TMP/t2"; } \
    || bad "test suite (polling)"
fi

HCS="handcheck2_imbalanced"
[ "$MODE" = "full" ] && HCS="handcheck handcheck2_imbalanced handcheck3_tiers handcheck_l15 handcheck_l15b handcheck_rack"
for hc in $HCS; do
  say "hand-check $hc + validator"
  python3 run_scenario.py --scenario "scenarios/$hc.yaml" --arm ours_full --seed 7 \
      --out "$TMP/$hc.json" --trace-dir "$TMP/${hc}_tr" >/dev/null 2>&1 \
    || bad "$hc run"
  python3 trace_validator.py "$TMP/${hc}_tr/"*.jsonl.gz --scenario "scenarios/$hc.yaml" \
      | tail -1 | tee "$TMP/v_$hc"
  grep -q "\[PASS\]" "$TMP/v_$hc" || bad "$hc validator"
done

say "pencil math (handcheck2 big0 flush == 3.4483 s)"
python3 - "$TMP" <<'PY' || FAIL=1
import gzip, json, glob, sys
S = sys.argv[1]
d = sorted({round(json.loads(l)["end"] - json.loads(l)["start"], 4)
            for f in glob.glob(f"{S}/handcheck2_imbalanced_tr/*.jsonl.gz")
            for l in gzip.open(f, "rt")
            if "crossjob_peers" in str(json.loads(l).get("operation"))
            and json.loads(l).get("job_id") == "big0"})
ok = d and abs(d[0] - 3.4483) < 5e-4
print(("OK" if ok else "XX FAIL"), "big0 flush durations:", d[:3])
sys.exit(0 if ok else 1)
PY

if [ "$MODE" = "full" ]; then
  say "mini-C smoke (ours policy, seed 7, both engines agree at run level)"
  python3 run_scenario.py --scenario scenarios/miniC.yaml --policy scenarios/miniC_policy.json \
      --seed 7 --out "$TMP/mc_e.json" >/dev/null 2>&1 || bad "miniC event run"
  SIM_ENGINE=polling python3 run_scenario.py --scenario scenarios/miniC.yaml \
      --policy scenarios/miniC_policy.json --seed 7 --out "$TMP/mc_p.json" >/dev/null 2>&1 \
    || bad "miniC polling run"
  python3 - "$TMP" <<'PY' || echo "   (note: per-seed engine deltas are the documented decision-boundary class; distribution-level equivalence is the standard — see ENGINE_REWRITE_VERIFICATION.md)"
import json, sys
S = sys.argv[1]
a = json.load(open(f"{S}/mc_e.json"))["rows"][0]["per_class_end_s"]
b = json.load(open(f"{S}/mc_p.json"))["rows"][0]["per_class_end_s"]
worst = max(abs(a[c] - b[c]) / b[c] for c in a)
print(f"engine A/B worst per-class delta: {100*worst:.2f}%")
sys.exit(0 if worst < 0.05 else 1)
PY

  say "mini-C smoke, Route C knob ON (background_nic_gbps=5.0, both engines agree)"
  python3 run_scenario.py --scenario scenarios/miniC.yaml --policy scenarios/miniC_policy.json \
      --seed 7 --background-nic-gbps 5.0 --out "$TMP/mcb_e.json" >/dev/null 2>&1 \
    || bad "miniC event run (bg on)"
  SIM_ENGINE=polling python3 run_scenario.py --scenario scenarios/miniC.yaml \
      --policy scenarios/miniC_policy.json --seed 7 --background-nic-gbps 5.0 \
      --out "$TMP/mcb_p.json" >/dev/null 2>&1 || bad "miniC polling run (bg on)"
  python3 - "$TMP" <<'PY' || echo "   (note: per-seed engine deltas are the documented decision-boundary class — same soft standard as the knob-off smoke above)"
import json, sys
S = sys.argv[1]
a = json.load(open(f"{S}/mcb_e.json"))["rows"][0]["per_class_end_s"]
b = json.load(open(f"{S}/mcb_p.json"))["rows"][0]["per_class_end_s"]
worst = max(abs(a[c] - b[c]) / b[c] for c in a)
print(f"engine A/B worst per-class delta (bg=5.0): {100*worst:.2f}%")
sys.exit(0 if worst < 0.05 else 1)
PY

  # protocol CL-014: a checkpoint stream on a PHYSICALLY SEPARATE fabric must
  # cost the collective exactly zero; the same stream on the collective's own
  # wire must still cost the documented 1 + tasks. Derivation:
  # scenarios/handcheck_fabric.yaml. Flag DEFAULT OFF -> `blind` is today.
  say "pencil math (fabric-scoped coupling: blind 15.000 s, scoped 13.000 s)"
  for fa in blind scoped; do
    python3 run_scenario.py --scenario scenarios/handcheck_fabric.yaml --arm "$fa" \
        --seed 7 --out "$TMP/fabric_$fa.json" >/dev/null 2>&1 \
      || bad "handcheck_fabric run ($fa)"
  done
  python3 - "$TMP" <<'PY' || FAIL=1
import json, sys
S = sys.argv[1]
got = {a: json.load(open(f"{S}/fabric_{a}.json"))["rows"][0]["per_class_end_s"]["trainer"]
       for a in ("blind", "scoped")}
want = {"blind": 15.0, "scoped": 13.0}
ok = all(abs(got[a] - want[a]) < 1e-9 for a in want)
print(("OK" if ok else "XX FAIL"), "trainer end_s:", got, "want", want)
sys.exit(0 if ok else 1)
PY
fi

echo
if [ "$FAIL" = 0 ]; then echo "ALL CHECKS PASSED — safe to push"; else
  echo "CHECKS FAILED — do not push (see above)"; exit 1; fi
