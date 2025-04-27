from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from predict import prediction
import csv
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import os
import gensim
import torch
from torch import nn

def save_to_csv(path, data):
    with open(path, mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(data)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class Item(BaseModel):
    title: str
    label: str

ws_driver = None
pos_driver = None
doc2vec_model = None
model = None

class NeuralNetwork(nn.Module):
  def __init__(self, input_dim, output_dim):
      super().__init__()
      self.stack = nn.Sequential(
        nn.Linear(input_dim, 32),  
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, output_dim)
      )
  def forward(self, x):
      logits = self.stack(x)
      return logits


tokens = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "../train/user-labeled-words.csv")


@app.on_event("startup")
async def startup_event():
    global ws_driver, pos_driver, doc2vec_model, model
    ws_driver = CkipWordSegmenter(model="bert-base")
    pos_driver = CkipPosTagger(model="bert-base")
    doc2vec_model = gensim.models.Doc2Vec.load("doc2vec_model_4.bin")

    input_dim = doc2vec_model.vector_size
    output_dim = 9
    model = NeuralNetwork(input_dim, output_dim).to("cpu")
    model.load_state_dict(torch.load("model_state_dict.pth"))
    model.eval()

@app.on_event("shutdown")
async def shutdown_event():
    global ws_driver, pos_driver
    ws_driver = None
    pos_driver = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.get("/api/model/prediction/")
async def get_prediction(title: str):
    sorted_result, text_tokens = prediction(title, ws_driver, pos_driver, doc2vec_model, model)
    global tokens
    tokens= text_tokens
    return sorted_result

@app.post("/api/model/feedback/")
async def send_feedback(item: Item):
    tokens.insert(0, item.label)
    save_to_csv(file_path, tokens)
    return {"ok": True}