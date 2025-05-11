from __future__ import annotations

import logging
import signal
import sys
from functools import wraps


def exit_handler(func):
    """Exit handling decorator on exception

    The main purpose is logging on keyboard interrupt exception
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.exception("Keyboard Interrupt")
            sys.exit(signal.SIGINT)
        except Exception as e:
            logger.exception(e)
            sys.exit(1)

    return wrapper
