import os
import sys
import logging
from tqdm import tqdm
from viser import uplot

# Color scheme for dark mode GUI
WH_LOGO = (152, 199, 60)
BACKGROUND = (0.15, 0.15, 0.15)
LANDMARK_ALL = (0.9, 0.9, 0.9)
LANDMARK_LOCAL = (1.0, 0.1, 0.1)
KEYFRAME = (0.0, 1.0, 0.0)
CAMERA = (0.7, 0.7, 1.0)
GRAPH = (0.7, 0.7, 1.0)
SPANNING_TREE = (1.0, 0.1, 1.0)
LOOP = (1.0, 0.1, 0.1)
TRAJECTORY = (0.1, 1.0, 0.1)

# Plot series for time profiling
_SERIES_BASE = uplot.Series()
_SERIES_BASE["show"] = True
_SERIES_BASE["width"] = 1.0

SERIES_TIME = uplot.Series()

SERIES_FRAME_READ = _SERIES_BASE.copy()
SERIES_FRAME_READ["label"] = "Frame Read"
SERIES_FRAME_READ["stroke"] = "darkgray"

SERIES_FRAME_SKIP = _SERIES_BASE.copy()
SERIES_FRAME_SKIP["label"] = "Frame Skip"
SERIES_FRAME_SKIP["stroke"] = "lightgray"

SERIES_IMAGE_VIEW = _SERIES_BASE.copy()
SERIES_IMAGE_VIEW["label"] = "Image View"
SERIES_IMAGE_VIEW["stroke"] = "gray"

SERIES_LANDMARKS = _SERIES_BASE.copy()
SERIES_LANDMARKS["label"] = "Landmarks"
SERIES_LANDMARKS["stroke"] = "yellow"

SERIES_DENSE_POINTS = _SERIES_BASE.copy()
SERIES_DENSE_POINTS["label"] = "Dense Points"
SERIES_DENSE_POINTS["stroke"] = "teal"

SERIES_KEYFRAME_GRAPH = _SERIES_BASE.copy()
SERIES_KEYFRAME_GRAPH["label"] = "Keyframe Graph"
SERIES_KEYFRAME_GRAPH["stroke"] = "orange"

SERIES_TRAJECTORY = _SERIES_BASE.copy()
SERIES_TRAJECTORY["label"] = "Trajectory"
SERIES_TRAJECTORY["stroke"] = "green"

SERIES_VISUALIZATION = _SERIES_BASE.copy()
SERIES_VISUALIZATION["label"] = "Visualization"
SERIES_VISUALIZATION["stroke"] = "blue"

SERIES_TRACKING = _SERIES_BASE.copy()
SERIES_TRACKING["label"] = "Tracking"
SERIES_TRACKING["stroke"] = "red"

SERIES_PROCESSING = _SERIES_BASE.copy()
SERIES_PROCESSING["label"] = "Processing Time"
SERIES_PROCESSING["stroke"] = "purple"

SERIES_RT_DEADLINE = _SERIES_BASE.copy()
SERIES_RT_DEADLINE["label"] = "Real-Time Deadline"
SERIES_RT_DEADLINE["stroke"] = "red"
SERIES_RT_DEADLINE["dash"] = (5,5)

RUNTIME_SERIES = (
    SERIES_TIME,
    SERIES_PROCESSING,
    SERIES_TRACKING,
    SERIES_VISUALIZATION,
    SERIES_FRAME_READ,
    SERIES_FRAME_SKIP,
    SERIES_IMAGE_VIEW,
    SERIES_LANDMARKS,
    SERIES_DENSE_POINTS,
    SERIES_KEYFRAME_GRAPH,
    SERIES_TRAJECTORY,
    SERIES_RT_DEADLINE,
)


# Colored Logging
_SPDLOG_LEVEL_TO_PY = {
    "trace": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "off": logging.CRITICAL + 10,
}

_LOG_LEVEL_COLOR = {
    logging.DEBUG:    "\033[36m",          # cyan
    logging.INFO:     "\033[32m",          # green
    logging.WARNING:  "\033[33m\033[1m",   # yellow bold
    logging.ERROR:    "\033[31m\033[1m",   # red bold
    logging.CRITICAL: "\033[1m\033[41m",   # bold on red
}

_TERMS_WITH_COLOR = (
    "ansi",
    "color",
    "console",
    "cygwin",
    "gnome",
    "konsole",
    "kterm",
    "linux",
    "msys",
    "putty",
    "rxvt",
    "screen",
    "vt100",
    "xterm",
)

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

class SpdLogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.color = has_color_support()

    def format(self, record):
        color_code = _LOG_LEVEL_COLOR.get(record.levelno)
        initial_name = record.levelname
        record.levelname = initial_name.lower()
        if self.color and color_code is not None:
            record.levelname = f"{color_code}{record.levelname}\033[0m"
        try:
            return super().format(record)
        finally:
            record.levelname = initial_name

def has_color_support(file=None) -> bool:
    fp = file if file is not None else sys.stdout
    if not hasattr(fp, "isatty") or not fp.isatty():
        return False
    term = os.environ.get("TERM", "")
    return any(t in term for t in _TERMS_WITH_COLOR)

def setup_logger(log_level: str) -> logging.Logger:
    log = logging.getLogger("stella_vslam")
    if not log.hasHandlers():
        handler = TqdmLoggingHandler()
        handler.setFormatter(SpdLogFormatter())
        log.addHandler(handler)
    log.setLevel(_SPDLOG_LEVEL_TO_PY[log_level.lower()])
    log.propagate = False
    return log
