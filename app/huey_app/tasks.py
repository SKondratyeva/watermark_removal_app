from model.unet_parts import *
import torch
from PIL import Image, ImageFilter
import numpy as np
import io
from huey import RedisHuey


path_model = "model/unet.pt"
model_dict = torch.load(path_model,  map_location=torch.device('cpu'))
model = Unet()
model.load_state_dict(model_dict)
model.eval()

#here the host is assigned the name of the redis service
# in the docker compose file
huey = RedisHuey('testing', host = 'redis')

@huey.task()
def predict_wm(image):

    image = Image.open(io.BytesIO(image))
    image = image.resize((256, 256))
    image = image.convert('L')

    image = image.filter(ImageFilter.FIND_EDGES)
    image = np.array(image)
    image = torch.tensor(image).type(torch.float).unsqueeze(0).reshape([1, 1, 256, 256])
    predicted_image = model(image).detach().numpy()[0][0] * 255
    predicted_image = predicted_image.astype(np.uint8)
    predicted_image = Image.fromarray(predicted_image)
    with io.BytesIO() as buf:
        predicted_image = predicted_image.convert('RGB')
        predicted_image.save(buf, format='JPEG')
        im_bytes = buf.getvalue()
    #the output is returned in str format and converted to bytes in the main file
    return im_bytes

