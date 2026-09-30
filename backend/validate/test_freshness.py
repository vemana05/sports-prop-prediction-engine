from backend.validate.freshness import compute_freshness

VALID_STATUSES = {"fresh", "stale", "never_run"}


def test_freshness_reports_every_registered_source(conn):
    freshness = compute_freshness(conn)
    assert freshness, "no sources registered in config/sources.yaml"
    for name, info in freshness.items():
        assert info["status"] in VALID_STATUSES, f"{name}: unexpected status {info['status']}"


def test_freshness_summary(conn, capsys):
    freshness = compute_freshness(conn)
    for name, info in sorted(freshness.items()):
        print(f"  {name}: {info['status']} (last success: {info['last_success']})")
