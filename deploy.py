#!/usr/bin/env python3
"""
Deployment Helper Script for Edu Tutor AI
This script helps validate configuration and prepare for deployment
"""

import os
import sys
import subprocess
import requests
from urllib.parse import urlparse

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print("✅ Python version is compatible")
    return True

def check_required_files():
    """Check if all required files exist"""
    required_files = ['app.py', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file} exists")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_secrets_config():
    """Check if secrets configuration exists"""
    secrets_path = ".streamlit/secrets.toml"
    if not os.path.exists(secrets_path):
        print("⚠️  secrets.toml not found. Creating template...")
        os.makedirs(".streamlit", exist_ok=True)
        
        template = """# Streamlit secrets configuration
MONGODB_URI = "your_mongodb_connection_string_here"
HF_TOKEN = "your_hugging_face_token_here"
IBM_API_KEY = "your_ibm_api_key_here"  # Optional
IBM_PROJECT_ID = "your_ibm_project_id_here"  # Optional
"""
        with open(secrets_path, 'w') as f:
            f.write(template)
        print(f"📝 Created template at {secrets_path}")
        print("Please update the configuration with your actual credentials")
        return False
    else:
        print("✅ secrets.toml exists")
        return True

def validate_mongodb_uri(uri):
    """Basic validation of MongoDB URI"""
    if not uri or uri == "your_mongodb_connection_string_here":
        return False
    
    if not uri.startswith(("mongodb://", "mongodb+srv://")):
        return False
    
    try:
        parsed = urlparse(uri)
        if not parsed.hostname:
            return False
        return True
    except:
        return False

def validate_hf_token(token):
    """Basic validation of Hugging Face token"""
    if not token or token == "your_hugging_face_token_here":
        return False
    
    if not token.startswith("hf_"):
        return False
    
    return len(token) > 10

def test_hugging_face_connection(token):
    """Test connection to Hugging Face"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            "https://huggingface.co/api/whoami", 
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Hugging
