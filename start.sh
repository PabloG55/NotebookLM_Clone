#!/bin/bash
echo "Starting Application..."

# Activate virtual environment
source venv/bin/activate

# Launch FastAPI Backend in the background
echo "Starting FastAPI Backend..."
venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 &
PID_API=$!

# Launch Gradio Frontend in the foreground
echo "Starting Gradio Frontend..."
venv/bin/python gradio_app.py

# When Gradio exits, kill FastAPI
kill $PID_API
echo "Application stopped."
