FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=movie_recommendation.settings

EXPOSE 8000

CMD ["gunicorn", "movie_recommendation.wsgi:application", "--bind", "0.0.0.0:8000"]
