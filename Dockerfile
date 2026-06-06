FROM python:3.10-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Extra deps used only by the demo
RUN pip install --no-cache-dir flask plotly requests

# Copy only what's needed
COPY model/ /app/model/
COPY demo/ /app/demo/

# Writable dir for CSV data and prediction history
RUN mkdir -p /app/demo/data /app/demo/prediction_history

WORKDIR /app/demo

EXPOSE 7071

CMD ["gunicorn", "--bind", "0.0.0.0:7071", "app:app"]
