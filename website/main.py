from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from predict import prediction
import csv

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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.get("/api/model/prediction/")
async def get_prediction(title: str):
    return prediction(title)

@app.post("/api/model/feedback/")
async def send_feedback(item: Item):
    save_to_csv("user-labeled-titles.csv", [item.title, item.label])
    return {"ok": True}