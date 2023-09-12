def save_image(imagefield_data):
    # deconstruct the file into something celery can pickle and send to the worker
    data = {}
    # file is a list containing all info needed to recontruct a UploadedFile object, in the correct order into which they have to
    # be passed to __init__
    file_info = []
    # name of the form field
    file_info.append(imagefield_data.field_name)
    # name of the image
    file_info.append(imagefield_data.name)
    file_info.append(imagefield_data.content_type)
    file_info.append(imagefield_data.size)
    file_info.append(imagefield_data.charset)
    # the actual data of the image, read into a string
    data['data'] = imagefield_data.read()

    # send the image to be saved by a worker
    return data


def predict_wm(image):

    image = Image.open(io.BytesIO(image))
    image = np.array(image)
    image = transform_norm(image).unsqueeze(0)

    predicted_image = model(image)[0].detach().numpy() * 255

    print('output SHAPE', predicted_image.shape)
    predicted_image = predicted_image.astype(np.float64)[0]  # Change to float64
    print('after conversion', predicted_image.shape, predicted_image.dtype)
    pil_image = Image.fromarray(predicted_image.astype(np.uint8).transpose(1, 2, 0))
   # predicted_image = Image.fromarray(predicted_image.astype(np.uint8), mode = 'RGB')  # Convert to uint8 before creating Image object
    with io.BytesIO() as buf:
        predicted_image = pil_image.convert('RGB')
        predicted_image.save(buf, format='JPEG')
        im_bytes = buf.getvalue()
    return im_bytes