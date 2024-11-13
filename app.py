from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


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
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def preprocess(data):
    df = pd.DataFrame(data)

    def convert_to_float(value):
        try:
            return float(value)
        except ValueError:
            return np.nan

    df['mileage'] = df['mileage'].astype(str).str.split().str[0].apply(convert_to_float)
    df['engine'] = df['engine'].astype(str).str.split().str[0].apply(convert_to_float)
    df['max_power'] = df['max_power'].astype(str).str.split().str[0].apply(convert_to_float)

    df = df.drop(columns=['torque'], errors='ignore')
    df = df.drop(columns=['selling_price'], errors='ignore')

    df['engine'] = df['engine'].fillna(0).astype(int)
    df['seats'] = df['seats'].fillna(0).astype(int)

    df['brand'] = df['name'].apply(lambda x: x.split()[0])
    df['model'] = df['name'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else "Unknown")
    df = df.drop(columns=['name'])

    X_num = df.select_dtypes(exclude=['object']).drop(columns=['seats'], errors='ignore')
    X_obj = df.select_dtypes(include=['object'])
    X_obj['seats'] = df['seats'].astype(str)
    X_obj_encoded = encoder.transform(X_obj)

    X_final = pd.DataFrame(np.hstack([X_num.values, X_obj_encoded]), columns=np.concatenate([X_num.columns, encoder.get_feature_names_out()]))

    # Заполняем отсутствующие столбцы нулями
    missing_cols = set(best_model.feature_names_in_) - set(X_final.columns)
    for col in missing_cols:
        X_final[col] = 0

    # Упорядочиваем столбцы
    X_final = X_final[best_model.feature_names_in_]

    return X_final

@app.post("/predict_item")
def predict_item(item: Item) -> JSONResponse:
    data = preprocess([item.model_dump()])
    prediction = best_model.predict(data)
    return JSONResponse(content={"predicted_price": float(prediction[0])})

@app.post("/predict_items")
async def predict_items(items: Items):
    data_dicts = [item.model_dump() for item in items.objects]
    data = preprocess(data_dicts)

    predictions = best_model.predict(data)
    items_data = [item.model_dump() for item in items.objects]
    
    for i, item_data in enumerate(items_data):
        item_data["predicted_price"] = float(predictions[i])

    df = pd.DataFrame(items_data)

    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)

    return JSONResponse(content={"file": output_file})