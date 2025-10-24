# UI PY 
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from config import config
import requests
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OFAC SDN Search UI")
templates = Jinja2Templates(directory="templates")
API_URL = os.getenv("API_URL", "http://localhost:8000/query")

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    """Display the search form"""
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(
    request: Request, 
    name: str = Form(...),
    dob: Optional[str] = Form(None),
    birthplace: Optional[str] = Form(None)
):
    """Handle form submission and search"""
    
    # Log the search request
    logger.info(f"Search request - Name: {name}, DOB: {dob}, Birthplace: {birthplace}")
    
    # Prepare payload with additional search parameters
    payload = {
        "query": name.strip(),
        "dob": dob.strip() if dob else None,
        "birthplace": birthplace.strip() if birthplace else None
    }
    
    # Remove empty values
    payload = {k: v for k, v in payload.items() if v}
    
    try:
        logger.info(f"Sending request to API: {payload}")
        resp = requests.post(API_URL, json=payload, timeout=300000)
        resp.raise_for_status()  # Raise an exception for bad status codes
        result = resp.json()
        
        # Add search parameters to result for display
        result["search_params"] = {
            "name": name,
            "dob": dob if dob else "",
            "birthplace": birthplace if birthplace else ""
        }
        
        logger.info(f"API response received - Decision: {result.get('decision', 'Unknown')}")
        
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timeout: {e}")
        result = {
            "error": "Request timed out. The translation service might be slow or unavailable.",
            "search_params": {"name": name, "dob": dob or "", "birthplace": birthplace or ""}
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        result = {
            "error": f"Connection error: {str(e)}",
            "search_params": {"name": name, "dob": dob or "", "birthplace": birthplace or ""}
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        result = {
            "error": f"Unexpected error: {str(e)}",
            "search_params": {"name": name, "dob": dob or "", "birthplace": birthplace or ""}
        }
    
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test API connectivity
        resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
        api_status = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        api_status = "unreachable"
    
    return {
        "status": "healthy",
        "api_status": api_status,
        "message": "OFAC SDN Search UI is running"
    }