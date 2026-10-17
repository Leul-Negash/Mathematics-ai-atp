"""Opt-in real Lean checks are required, not skipped, in the designated run."""


def pytest_addoption(parser):
    parser.addoption("--run-pantograph-live", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "pantograph_live: actual custom Lean executable integration"
    )
