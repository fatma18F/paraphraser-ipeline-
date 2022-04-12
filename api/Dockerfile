FROM python:3.8
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install SentencePiece
RUN pip install "fastapi[all]"
COPY ./app /code/app
COPY ./app/Flora_T5 /code/Flora_T5
ENV PYTHONPATH "${PYTHONPATH}:/code/app"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]