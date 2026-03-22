# COCO class IDs relevant to traffic
YOLO_CLASS_MAP = {
    # person (0) deliberately NOT included — excluded at detection level
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Road vehicles only — person excluded entirely
VEHICLE_GROUPS = {
    "CAR":       [2],       # car
    "BUS_TRUCK": [5, 7],    # bus + truck
    "BIKE":      [1, 3],    # bicycle + motorcycle only
}

# Feature names used by the ML model
ML_FEATURES = ["car_count", "bus_truck_count", "bike_count", "rain"]

# Green time classification classes (seconds)
GREEN_TIME_CLASSES = [30, 60, 90, 120]
CLASS_BOUNDARIES = {
    30:  (0,   45),
    60:  (45,  75),
    90:  (75,  105),
    120: (105, float("inf")),
}

MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 120
