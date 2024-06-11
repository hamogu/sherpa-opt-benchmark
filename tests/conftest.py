# Licensed under GLPv3 - see LICENSE

import logging
from pathlib import Path

from pytest import fixture

# logging configuration
# exec_log = logging.getLogger("algo")


@fixture(autouse=True)
def configure_logging(request, caplog):
    """Set log file name same as test name, and set log level."""
    output_dir = (
        Path(__file__).parent.parent
        / "results"
        / request.node.name.replace("[", "_").replace("]", "")
    )
    log_file = output_dir / ("log.log")
    request.config.pluginmanager.get_plugin("logging-plugin").set_log_path(log_file)
    caplog.set_level(logging.INFO)