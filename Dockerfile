FROM python:3.10
RUN pip install python-multipart
RUN pip install uvicorn
RUN pip install Jinja2
RUN pip install fastapi
RUN pip install pandas
RUN pip install numpy
RUN pip install pillow
RUN pip install torch
RUN pip install ormar
RUN pip install asyncpg
RUN pip install psycopg2-binary
RUN pip install python-dotenv
RUN pip install asyncpg
RUN pip install python-dotenv
RUN pip install pydantic


WORKDIR /app
COPY ./app /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
