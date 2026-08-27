"""File-and-terminal logging.

Every record reaches the log file. A record reaches the terminal only if it
carries `extra=TO_BOTH`, which `ConsoleFilter` is the gate on.
"""

import logging
import sys

TO_BOTH = {"console": True}   # extra= for a record that also prints

# this project's loggers. Everything else inherits the root's level, which is
# where third-party DEBUG chatter is capped.
OURS = ('hilo', 'pypolar', 'tablet', '__main__')

logger = logging.getLogger(__name__)


class ConsoleFilter(logging.Filter):
    def filter(self, record):
        return getattr(record, "console", False)


def setup_logger(logfile="app.log", level=logging.DEBUG,
                 noisy_level=logging.WARNING):
    """Attaches a file handler and a filtered console handler to the root logger.

    Replaces the handlers a previous call installed, so calling it twice does
    not duplicate every line.

    Args:
        logfile: the file every record is written to.
        level: the level the `OURS` loggers record at.
        noisy_level: the level every other logger, third-party included, is
            capped at.

    Returns: the (file handler, console handler) pair.
    """
    root = logging.getLogger()
    root.setLevel(noisy_level)
    for name in OURS:
        logging.getLogger(name).setLevel(level)

    for handler in list(root.handlers):
        if getattr(handler, '_pypolar', False):
            root.removeHandler(handler)
            handler.close()

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.NOTSET)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.NOTSET)
    ch.addFilter(ConsoleFilter())
    ch.setFormatter(logging.Formatter("%(message)s"))

    for handler in (fh, ch):
        handler._pypolar = True
        root.addHandler(handler)

    return fh, ch


def logged_input(prompt):
    """Prompts, and records the prompt and the answer to the file.

    `input` already echoes both to the terminal, so neither is marked console.
    """
    logger.info("PROMPT: %s", prompt)
    answer = input(prompt)
    logger.info("INPUT:  %s", answer)
    return answer
