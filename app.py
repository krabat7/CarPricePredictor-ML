from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI()

# Загрузка обученной модели Ridge
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Загрузка кодировщика
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int = 0
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str = ""
    seats: float


def preprocess(data: pd.DataFrame):
    def convert_to_float(value):
        try:
            return float(value)
        except ValueError:
            return np.nan

    data['mileage'] = data['mileage'].astype(str).str.split().str[0].apply(convert_to_float)
    data['engine'] = data['engine'].astype(str).str.split().str[0].apply(convert_to_float)
    data['max_power'] = data['max_power'].astype(str).str.split().str[0].apply(convert_to_float)

    data = data.drop(columns=['torque', 'selling_price'], errors='ignore')

    data['engine'] = data['engine'].fillna(0).astype(int)
    data['seats'] = data['seats'].fillna(0).astype(int)

    data['brand'] = data['name'].apply(lambda x: x.split()[0])
    data['model'] = data['name'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else "Unknown")
    data = data.drop(columns=['name'])

    X_num = data.select_dtypes(exclude=['object']).drop(columns=['seats'], errors='ignore')
    X_obj = data.select_dtypes(include=['object'])
    X_obj['seats'] = data['seats'].astype(str)

    X_obj_encoded = encoder.transform(X_obj)
    X_final = pd.DataFrame(
        np.hstack([X_num.values, X_obj_encoded]),
        columns=np.concatenate([X_num.columns, encoder.get_feature_names_out()])
    )

    # Заполняем отсутствующие столбцы нулями
    missing_cols = set(best_model.feature_names_in_) - set(X_final.columns)
    for col in missing_cols:
        X_final[col] = 0
        
    # Упорядочиваем столбцы
    X_final = X_final[best_model.feature_names_in_]
    return X_final


@app.post("/predict_item")
def predict_item(item: Item):
    data = preprocess(pd.DataFrame([item.dict()]))
    prediction = best_model.predict(data)
    return JSONResponse(content={"predicted_price": float(prediction[0])})


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    data = preprocess(df)
    predictions = best_model.predict(data)

    df['predicted_price'] = predictions

    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)

    return FileResponse(output_file, media_type="text/csv", filename="predictions.csv")
