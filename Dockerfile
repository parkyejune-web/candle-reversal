FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY live.py telegram_bot.py ./

CMD ["python", "-u", "live.py"]
