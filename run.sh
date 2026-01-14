# Save this as run.sh and make executable:
chmod +x run.sh

# Content of run.sh:
#!/bin/bash
echo "Starting ResearchMate..."
echo "1. Starting backend on http://localhost:8000"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
echo "2. Starting frontend on http://localhost:8501"
streamlit run frontend/app.py