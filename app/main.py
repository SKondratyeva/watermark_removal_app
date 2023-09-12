from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import base64
from huey_app.tasks import huey, predict_wm, image_preprocessing, convert_prediction
import torchvision.transforms as T

huey = huey
transform_norm=T.Compose([T.ToTensor()])


fastapi_app = FastAPI()
templates = Jinja2Templates(directory = "templates")


@fastapi_app.get('/', response_class = HTMLResponse)
async def home(request: Request):
    data = {
        'page':'home page'
    }
    return templates.TemplateResponse('page1.html', {'request': request, 'data': data})


@fastapi_app.post("/submitform", response_class = HTMLResponse )
async def handle_form(request: Request, image_upload: UploadFile = Form(...)):
    file_name = str(image_upload.filename)
    image = await image_upload.read()
    preprocessed_image = image_preprocessing(image)
    preprocessed_image = preprocessed_image.get(blocking=True)
    predicted_tensor = predict_wm(preprocessed_image)
    predicted_tensor = predicted_tensor.get(blocking=True)
    res = convert_prediction(predicted_tensor)
    res = res.get(blocking=True)
    res = base64.b64encode(res).decode("utf-8")
    orig = base64.b64encode(image).decode("utf-8")

    data = {
        "page": "Original image vs watermark-free image: ",
        "img_orig": orig,
        "img_proc": res
    }


    return templates.TemplateResponse("page1.html", {"request": request, "data": data})



@fastapi_app.get('/home', response_class = HTMLResponse)
async def home(request: Request):
    data = {
        'page':'home page'
    }
    return templates.TemplateResponse('page1.html', {'request': request, 'data': data})

@fastapi_app.get("/{page_name}", response_class=HTMLResponse)
async def page(request: Request, page_name: str):
    data = {
        "page": page_name
    }
    return templates.TemplateResponse("page1.html", {"request": request, "data": data})


