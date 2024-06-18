FROM python:3.9-slim

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user:user requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user:user . .

USER user

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
