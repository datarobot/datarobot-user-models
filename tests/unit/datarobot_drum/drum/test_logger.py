import logging
import uuid

from datarobot_drum.drum.common import ctx_request_id
from datarobot_drum.drum.common import get_drum_logger
from datarobot_drum.drum.common import request_id_filter


def run_logging_function():
    logger = get_drum_logger("test")
    logger.addFilter(request_id_filter)
    logger.warning("test message")


def test_drum_logger(caplog):
    sample_request_id = str(uuid.uuid4())
    ctx_request_id.set(sample_request_id)
    with caplog.at_level(logging.WARNING):
        run_logging_function()
    log_record = caplog.records[0]
    assert log_record.name == "drum.test"
    assert log_record.message == "test message"
    assert log_record.request_id == sample_request_id
