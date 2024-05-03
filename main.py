from fastapi import FastAPI
from transformers import pipeline
import requests
from PIL import Image
from urllib.parse import unquote
# API_URL = "https://api-inference.huggingface.co/models/giacomoarienti/nsfw-classifier"
# headers = {"Authorization": "Bearer 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"}
app = FastAPI()
nlp = pipeline("sentiment-analysis", model='akhooli/xlm-r-large-arabic-toxic')
classifier = pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector")
@app.get('/')
async def root():
    return {'example': 'This is an example', 'data': 10}

@app.get('/validate-text')
async def validate_text(text: str):
    result = nlp(text)
    return {'result': result}

@app.get('/validate-image')
async def validate_image(url: str):
    image = Image.open(requests.get(url, stream=True).raw)
    result = classifier(image)
    return {'result': result}
