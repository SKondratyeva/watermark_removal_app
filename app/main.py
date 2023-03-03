from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import base64
from PIL import Image, ImageFilter
import io
import torch
import numpy as np
from model.unet_parts import *
from db import Database

database = Database()
app = FastAPI()
path_model = "model/unet.pt"
model_dict = torch.load(path_model,  map_location=torch.device('cpu') )
model = Unet()
model.load_state_dict(model_dict)
model.eval()
templates = Jinja2Templates(directory = "templates")

@app.get('/', response_class = HTMLResponse)
async def home(request: Request):
    data = {
        'page':'home page'
    }
    return templates.TemplateResponse('page.html', {'request': request, 'data': data})

@app.on_event("startup")
async def startup():
    database.connect()


@app.on_event("shutdown")
async def shutdown():
    database.disconnect()

# response with an HTML page when get requests is sent
@app.get('/home', response_class = HTMLResponse)
async def home(request: Request):
    data = {
        'page':'home page'
    }
    return templates.TemplateResponse('page.html', {'request': request, 'data': data})

@app.get("/{page_name}", response_class=HTMLResponse)
async def page(request: Request, page_name: str):
    data = {
        "page": page_name
    }
    return templates.TemplateResponse("page.html", {"request": request, "data": data})


@app.post("/submitform", response_class=HTMLResponse)
async def handle_form(request: Request, image_upload: UploadFile = Form(...)):
    file_name = str(image_upload.filename)
    image = await image_upload.read()
    image = Image.open(io.BytesIO(image))
    database.write(file_name)
    image = image.resize((256, 256))
    image = image.convert('L')
    image = image.filter(ImageFilter.FIND_EDGES)
    image = np.array(image)
    image = torch.tensor(image).type(torch.float).unsqueeze(0).reshape([1, 1, 256, 256])
    predicted_image = model(image).detach().numpy()[0][0]*255
  #  predicted_image = predicted_image.astype(np.uint8)
    predicted_image = Image.fromarray(predicted_image)


   # predicted_image = base64.b64encode(predicted_image).decode("utf-8")
    with io.BytesIO() as buf:
        predicted_image = predicted_image.convert('RGB')
        predicted_image.save(buf, format='JPEG')
        im_bytes = buf.getvalue()
        im_bytes = base64.b64encode(im_bytes).decode("utf-8")
    data = {
        "page": "submission",
        "img":im_bytes
    }


    return templates.TemplateResponse("page.html", {"request": request, "data": data})
