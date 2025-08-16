import joblib, pandas as pd

def load_pipeline(path="artifacts/model.joblib"):
    return joblib.load(path)
def friendly_to_model_inputs(user_inputs: dict) -> dict:
    """
    Convert user-friendly inputs into model-ready features.
    """
    return {
        "MedInc": user_inputs["income"] / 10000,   # convert dollars to 10k units
        "HouseAge": user_inputs["house_age"],
        "AveRooms": user_inputs["rooms"],
        "AveBedrms": user_inputs["bedrooms"],
        "Population": user_inputs["population"],
        "AveOccup": user_inputs["occupants"],
        "Latitude": user_inputs["latitude"],
        "Longitude": user_inputs["longitude"],
    }

def predict_one(pipeline, features: dict, task="regression", target="MedHouseVal"):
    import pandas as pd
    X = pd.DataFrame([features])
    y_pred = pipeline.predict(X)

    # Convert California Housing predictions to dollars
    if target == "MedHouseVal" and task == "regression":
        return y_pred[0] * 100000
    return y_pred[0]

# from src.utils import load_pipeline, predict_one
# pipe = load_pipeline()
# sample = {"MedInc": 3.5, "HouseAge": 15, "AveRooms": 5.2, "AveBedrms": 1.0,
#           "Population": 800, "AveOccup": 2.8, "Latitude": 34.1, "Longitude": -118.2}
# print(predict_one(pipe, sample))
