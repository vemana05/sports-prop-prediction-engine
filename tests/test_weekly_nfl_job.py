"""scripts/weekly_nfl_job.sh: waits for DNS before any OddsPapi step and
retries the Wednesday lines step once on a network error. Runs a copy of the
script against fake `docker` and `sleep` commands; nothing real is started."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "weekly_nfl_job.sh"

# Fake docker: logs each call, answers the DNS probe from $DNS_OK_AFTER (number
# of failed probes before it resolves), and plays the OddsPapi step's outcomes
# from $ODDS_RESULTS ("neterr,ok" = network error, then success).
FAKE_DOCKER = r"""#!/bin/bash
echo "$*" >> "$CALLS"
case "$*" in
  info) exit 0 ;;
  "compose ps"*) exit 0 ;;
  *socket.getaddrinfo*)
    n=$(grep -c getaddrinfo "$CALLS")
    [ "$n" -gt "${DNS_OK_AFTER:-0}" ] && exit 0 || exit 1 ;;
  *odds_oddspapi_nfl*)
    n=$(grep -c odds_oddspapi_nfl "$CALLS")
    r=$(echo "$ODDS_RESULTS" | cut -d, -f"$n")
    case "$r" in
      ok) echo '{"games": 14, "requests": 15}'; exit 0 ;;
      neterr) echo "{\"error\": \"HTTPSConnectionPool(host='api.oddspapi.io', port=443): Max retries exceeded (NameResolutionError)\"}"; exit 1 ;;
      *) echo '{"error": "OddsPapi /odds -> 401: bad key"}'; exit 1 ;;
    esac ;;
  *) exit 0 ;;
esac
"""


@pytest.fixture
def job(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", FAKE_DOCKER), ("sleep", "#!/bin/bash\necho \"sleep $*\" >> \"$CALLS\"\n")):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / "calls.txt"

    def run(mode="lines", odds="ok", dns_ok_after=0):
        calls.write_text("")
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CALLS": str(calls),
               "ODDS_RESULTS": odds, "DNS_OK_AFTER": str(dns_ok_after)}
        p = subprocess.run(["bash", str(tmp_path / "scripts" / SCRIPT.name), mode], env=env,
                           capture_output=True, text=True, timeout=60)
        log = "".join(f.read_text() for f in sorted((tmp_path / "logs").glob(f"weekly_nfl_job_{mode}_*.log")))
        for f in (tmp_path / "logs").glob("*.log"):
            f.unlink()
        lines = calls.read_text().splitlines()
        return p.returncode, lines, log

    return run


def _odds_calls(lines):
    return [ln for ln in lines if "odds_oddspapi_nfl" in ln]


def test_lines_waits_for_dns_then_collects(job):
    code, calls, log = job(dns_ok_after=3)
    assert code == 0
    assert len(_odds_calls(calls)) == 1
    assert sum("sleep 60" in ln for ln in calls) == 3
    # No OddsPapi request before DNS resolved.
    first_odds = next(i for i, ln in enumerate(calls) if "odds_oddspapi_nfl" in ln)
    assert sum("getaddrinfo" in ln for ln in calls[:first_odds]) == 4
    assert "no network yet (attempt 3/30)" in log


def test_lines_retries_once_on_network_error(job):
    code, calls, log = job(odds="neterr,ok")
    assert code == 0 and len(_odds_calls(calls)) == 2
    assert "failed on a network error; retrying once" in log


def test_lines_retry_is_only_once(job):
    code, calls, _ = job(odds="neterr,neterr,ok")
    assert code == 1 and len(_odds_calls(calls)) == 2


def test_lines_does_not_retry_other_errors(job):
    code, calls, log = job(odds="apierr,ok")
    assert code == 1 and len(_odds_calls(calls)) == 1
    assert "retrying once" not in log


def test_no_network_spends_nothing(job):
    code, calls, log = job(dns_ok_after=99)
    assert code == 1 and _odds_calls(calls) == []
    assert sum("getaddrinfo" in ln for ln in calls) == 30
    assert "ERROR: no network after 30 min" in log


def test_results_step_waits_but_does_not_retry(job):
    code, calls, log = job(mode="results", odds="neterr,ok")
    assert code == 0  # stats and features are in; the closing-line failure is a warning
    assert len(_odds_calls(calls)) == 1 and "retrying once" not in log
    assert "WARNING: OddsPapi closing-line step failed" in log
