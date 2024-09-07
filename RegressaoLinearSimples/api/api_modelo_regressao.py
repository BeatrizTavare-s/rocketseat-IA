from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar uma instancia do FastAPI
app =FastAPI()

# Criar um class que terá os dados do request body
class request_body(BaseModel):
    horas_estudo: float

# Carregar modelo para realizar a predição
modelo_pontuacao = joblib.load('../modelo_pregressao.pkl')

@app.post('/predict')
def predct(data: request_body):
    # Preparar os dados para predição
    input_feature = [[data.horas_estudo]]

    # Realizar a predição
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

    return {"pontuacao_test": y_pred.tolist()}