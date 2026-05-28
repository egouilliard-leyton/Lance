from live_meeting.config import LiveMeetingConfig, parse_config


def test_defaults():
    c = parse_config([])
    assert c.mode == "simulate"
    assert c.engine == "fake"
    assert c.fps == 1.0
    assert c.audio_sample_rate == 16000
    assert c.realtime is True
    assert c.output_path == "results/live_meeting"


def test_overrides():
    c = parse_config(["--engine", "ixc", "--fps", "2", "--mode", "eval"])
    assert c.engine == "ixc"
    assert c.fps == 2.0
    assert c.mode == "eval"


def test_no_realtime_flag():
    assert parse_config(["--no-realtime"]).realtime is False


def test_recall_key_from_env(monkeypatch):
    monkeypatch.setenv("RECALL_API_KEY", "secret-123")
    assert parse_config([]).recall_api_key == "secret-123"


def test_dataclass_default_reads_env(monkeypatch):
    monkeypatch.setenv("RECALL_API_KEY", "abc")
    assert LiveMeetingConfig().recall_api_key == "abc"
