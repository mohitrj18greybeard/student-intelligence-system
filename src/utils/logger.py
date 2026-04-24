"""UTF-8 safe logger for EduPulse AI."""
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"edupulse.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        try:
            handler.stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        fmt = logging.Formatter(
            "%(asctime)s │ %(name)-30s │ %(levelname)-8s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
