"""scripts/nba_gamelog_retry_job.sh (1 PM): re-runs only the NBA game-log
step, only when that morning's daily job left its failure marker, and only
once. Runs a copy of the script against fake `docker` and `caffeinate`
commands; nothing real is started."""

import os
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nba_gamelog_retry_job.sh"

FAKE_DOCKER = r"""#!/bin/bash
echo "$*" >> "$CALLS"
case "$*" in
  *nba_daily_games*) echo "loaded"; exit "${GAMELOG_EXIT:-0}" ;;
  *) exit 0 ;;
esac
"""
FAKE_CAFFEINATE = "#!/bin/bash\necho \"caffeinate $*\" >> \"$CALLS\"\nshift\nexec \"$@\"\n"


@pytest.fixture
def job(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "logs").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / SCRIPT.name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", FAKE_DOCKER), ("caffeinate", FAKE_CAFFEINATE)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / "calls.txt"
    marker = tmp_path / "logs" / f"nba_gamelogs_retry_{date.today():%Y%m%d}"

    def run(failed_this_morning, gamelog_exit=0):
        calls.write_text("")
        if failed_this_morning:
            marker.touch()
        env = {k: v for k, v in os.environ.items() if k != "PARLAY_CAFFEINATED"}
        env = {**env, "PATH": f"{bin_dir}:{env['PATH']}", "CALLS": str(calls), "GAMELOG_EXIT": str(gamelog_exit)}
        p = subprocess.run(["bash", str(tmp_path / "scripts" / SCRIPT.name)], env=env,
                           capture_output=True, text=True, timeout=60)
        log = "".join(f.read_text() for f in sorted((tmp_path / "logs").glob("nba_gamelog_retry_*.log")))
        for f in (tmp_path / "logs").glob("*.log"):
            f.unlink()
        return p.returncode, calls.read_text().splitlines(), log, marker.exists()

    return run


def test_nothing_to_retry_when_the_morning_step_succeeded(job):
    code, calls, log, _ = job(failed_this_morning=False)
    assert code == 0
    assert [ln for ln in calls if not ln.startswith("caffeinate")] == []  # Docker untouched
    assert "nothing to retry" in log and "(gamelogs exit 0)" in log


def test_retries_only_the_game_log_step_once(job):
    code, calls, log, marker_left = job(failed_this_morning=True)
    assert code == 0 and not marker_left
    runs = [ln for ln in calls if ln.startswith("compose run")]
    assert len(runs) == 1 and "backend.ingest.nba_daily_games" in runs[0]
    assert not any("log_predictions" in ln or "odds" in ln or "schedule_projections" in ln for ln in calls)
    assert calls[0].startswith("caffeinate -i ")
    # Second firing the same day: marker gone, nothing re-run.
    _, calls, _, _ = job(failed_this_morning=False)
    assert not any("nba_daily_games" in ln for ln in calls)


def test_failed_retry_reports_the_exit_and_does_not_retry_again(job):
    code, _, log, marker_left = job(failed_this_morning=True, gamelog_exit=1)
    assert code == 1 and not marker_left
    assert "(gamelogs exit 1)" in log
