FROM python:3.9-slim

WORKDIR /app

# copy only the requirements.txt first to leverage Docker cache
COPY ./app/requirements.txt /app/requirements.txt

#install the libraries
RUN pip install -r /app/requirements.txt

# copy the rest of the application
COPY ./app /app

EXPOSE 8888

CMD ["uvicorn", "main:fastapi_app", "--host", "0.0.0.0", "--port", "8888"]
