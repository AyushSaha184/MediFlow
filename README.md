# MediFlow

A production-grade multi-agent Medical AI system.

## Setup Instructions

1. **Install Dependencies**
   It's recommended to use a virtual environment.

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows PowerShell
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   Copy the `.env` template or set variables manually. For dev no `.env` is strictly required as default values are set.

   Example `.env` content:

   ```
   ENVIRONMENT=dev
   LOG_LEVEL=INFO
   ```

3. **Run the Application**

   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Verify it's running**
   Open a browser or use cURL:

   ```bash
   curl http://localhost:8000/health
   # Expected Output JSON: {"status": "ok", "environment": "dev"}
   ```
