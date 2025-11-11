import pickle
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI

# Note that we find the pipeline file!
model_file = "seeds_classifier.pkl"


with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)


app = FastAPI(title="predict_api_service")


@app.post("/predict")
def predict(input: dict[str, Any]):
    df = pd.Series(input).to_frame().T
    y_pred_probability = model.predict_proba(df)[0]
    y_pred_class = model.predict(df)[0]
    result = {
        "variety": int(y_pred_class),
        "variety_probability": float(y_pred_probability[y_pred_class]),
    }

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
