import json
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from api.model import Input
from app import app, logger
import torch
import pandas as pd
import xgboost as xgb
model_api = APIRouter()


def get_bert_embeddings(texts):
    # Tokenize and prepare inputs
    encoded_input = app.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128).to("cuda")
    with torch.no_grad():
        # Forward pass, get hidden states
        output = app.bert(**encoded_input)

    # Extract embeddings from the last hidden state
    embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def get_time_features(created_at):
    # Extract day of the week and hour
    df = pd.DataFrame()
    df['Created_at'] = pd.to_datetime([created_at])

    df['Day_of_Week'] = df['Created_at'].dt.dayofweek
    df['Hour'] = df['Created_at'].dt.hour
    # One-hot encode the day of the week
    df_one_hot = pd.get_dummies(df['Day_of_Week'], prefix='Day')
    df = pd.concat([df, df_one_hot], axis=1)
    return df.drop(columns=['Created_at', 'Day_of_Week'])

@model_api.post("/predict")
def predict(data: Input) -> JSONResponse:
    embs = get_bert_embeddings([data.Comment])  # Input should be a list of comments
    time_features = get_time_features(data.Created_at)

    # Convert Input data to DataFrame
    df = pd.DataFrame([data.dict()])

    # Remove non-feature columns if necessary (e.g., Comment, Created_at)
    df = df.drop(columns=['Comment', 'Created_at'])

    # Concatenate the embeddings and time features
    embed_df = pd.DataFrame(embs, columns=[f'emb_{i}' for i in range(embs.shape[1])])
    df = pd.concat([df.reset_index(drop=True), embed_df.reset_index(drop=True)], axis=1)
    df = pd.concat([df, time_features.reset_index(drop=True)], axis=1)

    # Prepare DMatrix for prediction
    ddata = xgb.DMatrix(df[app.model.features_in_])

    # Get prediction probabilities
    pred_probs = app.model.predict(ddata)

    return JSONResponse(
        content={"output": pred_probs.tolist()}, status_code=200
    )
