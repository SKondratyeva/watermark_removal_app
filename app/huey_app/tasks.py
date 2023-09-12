from model.unet_parts import *
import torch
from PIL import Image, ImageFilter
import numpy as np
import io
from huey import RedisHuey
from model.unet import *
import torchvision.transforms as T

path_model = "model/model.pth"
model_dict = torch.load(path_model,  map_location=torch.device('cpu'))
model = generator(3, 3)
model.load_state_dict(model_dict)
model.eval()


huey = RedisHuey('myapp', host='redis', port=6379)

@huey.task()
def image_preprocessing(image):
    transform_norm = T.Compose([T.ToTensor()])
    image = Image.open(io.BytesIO(image))
    image = np.array(image)
    image = transform_norm(image).unsqueeze(0)
    return image


@huey.task()
def predict_wm(image):
    predicted_image = model(image)
    return predicted_image


@huey.task()
def convert_prediction(prediction):
    predicted_image = prediction[0].detach().numpy() * 255
    predicted_image = predicted_image.astype(np.float64)[0]  # Change to float64

    pil_image = Image.fromarray(predicted_image.astype(np.uint8).transpose(1, 2, 0))
    with io.BytesIO() as buf:
        predicted_image = pil_image.convert('RGB')
        predicted_image.save(buf, format='JPEG')
        im_bytes = buf.getvalue()
    return im_bytes