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
SERIES_BASE = uplot.Series()
SERIES_BASE["show"] = True
SERIES_BASE["width"] = 1.0

SERIES_TIME = uplot.Series()

SERIES_IMAGE_VIEW = SERIES_BASE.copy()
SERIES_IMAGE_VIEW["label"] = "Image"
SERIES_IMAGE_VIEW["stroke"] = "gray"

SERIES_LANDMARKS = SERIES_BASE.copy()
SERIES_LANDMARKS["label"] = "Landmark"
SERIES_LANDMARKS["stroke"] = "yellow"

SERIES_DENSE_POINTS = SERIES_BASE.copy()
SERIES_DENSE_POINTS["label"] = "Dense Points"
SERIES_DENSE_POINTS["stroke"] = "teal"

SERIES_KEYFRAME_GRAPH = SERIES_BASE.copy()
SERIES_KEYFRAME_GRAPH["label"] = "Keyframe Graph"
SERIES_KEYFRAME_GRAPH["stroke"] = "orange"

SERIES_TRAJECTORY = SERIES_BASE.copy()
SERIES_TRAJECTORY["label"] = "Trajectory"
SERIES_TRAJECTORY["stroke"] = "green"

SERIES_VISUALIZATION = SERIES_BASE.copy()
SERIES_VISUALIZATION["label"] = "Visualization"
SERIES_VISUALIZATION["stroke"] = "blue"

SERIES_TRACKING = SERIES_BASE.copy()
SERIES_TRACKING["label"] = "Tracking"
SERIES_TRACKING["stroke"] = "red"

SERIES_PROCESSING = SERIES_BASE.copy()
SERIES_PROCESSING["label"] = "Processing Time"
SERIES_PROCESSING["stroke"] = "purple"

SERIES_RT_DEADLINE = SERIES_BASE.copy()
SERIES_RT_DEADLINE["label"] = "Real-Time Deadline"
SERIES_RT_DEADLINE["stroke"] = "red"
SERIES_RT_DEADLINE["dash"] = (5,5)
