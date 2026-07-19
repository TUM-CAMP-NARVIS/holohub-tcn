from .app_window import MainWindowApp
from slint import quit_event_loop
import logging
__all__ = ["MainWindowApp", "close_controller_app", ]

log = logging.getLogger(__name__)


def close_controller_app(*args, **kw):
    log.info("Exit GUI Loop.")
    quit_event_loop()