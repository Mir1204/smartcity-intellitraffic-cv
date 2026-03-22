import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgb_green_time.joblib")

_model = None

# Map class indices to green time values
CLASS_MAP = {0: 30, 1: 60, 2: 90, 3: 120}


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_green_time_class(car_count, bus_truck_count, bike_count, rain=0):
    """
    Predict optimal green signal time class.
    
    Returns:
        dict with keys: predicted_class (int), class_label (str),
                       probabilities (dict), raw_class_index (int)
    """
    model = load_model()
    X = np.array([[car_count, bus_truck_count, bike_count, rain]])
    
    # Get the predicted class
    class_idx = int(model.predict(X)[0])
    
    # Get probabilities for all classes
    proba = model.predict_proba(X)[0]
    
    class_label = CLASS_MAP[class_idx]
    
    # Create probability dict
    probabilities = {
        f"class_{CLASS_MAP[i]}s": float(proba[i]) 
        for i in range(len(proba))
    }
    
    return {
        "predicted_class_index": class_idx,
        "predicted_green_time": class_label,
        "probabilities": probabilities,
        "confidence": float(proba[class_idx])
    }


def predict_green_time(car_count, bus_truck_count, bike_count, rain=0):
    """
    Simple wrapper that returns only the predicted green time value (30, 60, 90, or 120).
    
    Returns:
        int: Predicted green time in seconds (30, 60, 90, or 120)
    """
    result = predict_green_time_class(car_count, bus_truck_count, bike_count, rain)
    return result["predicted_green_time"]


if __name__ == "__main__":
    # Test examples
    result = predict_green_time_class(10, 3, 5, rain=0)
    print(f"Predicted green time class: {result['predicted_green_time']}s")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"All probabilities: {result['probabilities']}")
    
    result_rain = predict_green_time_class(10, 3, 5, rain=1)
    print(f"\nWith rain - Predicted green time class: {result_rain['predicted_green_time']}s")
    print(f"Confidence: {result_rain['confidence']:.4f}")
