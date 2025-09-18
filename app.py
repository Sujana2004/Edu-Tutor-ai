import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pymongo
from pymongo import MongoClient
import bcrypt
import requests
import json
import os
from huggingface_hub import InferenceClient
import time

try:
    import pymongo
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    st.warning("MongoDB not available - running in demo mode only")

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    st.warning("HuggingFace client not available - using fallback responses")

# Page configuration
st.set_page_config(
    page_title="Edu Tutor AI - Intelligent Student Engagement Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        margin-left: 2rem;
    }
    
    .ai-message {
        background-color: #e8f4f8;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_analytics' not in st.session_state:
    st.session_state.user_analytics = {
        'total_interactions': 0,
        'avg_sentiment': 0.5,
        'topics_discussed': [],
        'session_duration': 0,
        'start_time': datetime.now()
    }

# Configuration
class Config:
    MONGODB_URI = st.secrets.get("MONGODB_URI", "")
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# MongoDB connection
@st.cache_resource
def init_mongodb():
    try:
        if not Config.MONGODB_URI:
            return None
        client = MongoClient(Config.MONGODB_URI)
        db = client['edu_tutor_ai']
        # Test connection
        client.admin.command('ping')
        return db
    except Exception as e:
        st.error(f"MongoDB connection error: {e}")
        return None

# Initialize lightweight AI client
@st.cache_resource
def init_ai_client():
    try:
        if not Config.HF_TOKEN:
            st.warning("⚠️ Hugging Face token not found. AI features will be limited.")
            return None
        
        # Use Hugging Face Inference API instead of loading models locally
        client = InferenceClient(token=Config.HF_TOKEN)
        return client
    except Exception as e:
        st.warning(f"AI client initialization error: {e}")
        return None

# Simple rule-based sentiment analysis (lightweight alternative)
def analyze_sentiment_simple(text):
    positive_words = ['good', 'great', 'excellent', 'awesome', 'amazing', 'love', 'like', 'happy', 'excited', 'wonderful']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed', 'boring']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return 0.7, "😊 Positive"
    elif negative_count > positive_count:
        return 0.3, "😔 Negative"
    else:
        return 0.5, "😐 Neutral"

# API-based sentiment analysis using HF Inference API
def analyze_sentiment_api(hf_client, text):
    try:
        if not hf_client:
            return analyze_sentiment_simple(text)
        
        # Use HF Inference API for sentiment analysis
        result = hf_client.text_classification(
            text, 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        if result:
            # Map labels to scores
            label_mapping = {
                'LABEL_0': 0.2,  # Negative
                'LABEL_1': 0.5,  # Neutral
                'LABEL_2': 0.8   # Positive
            }
            
            top_result = max(result, key=lambda x: x['score'])
            sentiment_score = label_mapping.get(top_result['label'], 0.5)
            
            if sentiment_score > 0.6:
                sentiment_label = "😊 Positive"
            elif sentiment_score < 0.4:
                sentiment_label = "😔 Negative"
            else:
                sentiment_label = "😐 Neutral"
                
            return sentiment_score, sentiment_label
    except Exception as e:
        st.warning(f"API sentiment analysis failed: {e}")
        return analyze_sentiment_simple(text)

# AI Response Generation using HF Inference API
def generate_ai_response(hf_client, user_input, context=""):
    try:
        if not hf_client:
            # Fallback responses when AI is not available
            fallback_responses = [
                f"Thank you for your question about '{user_input[:50]}...'. As an educational tutor, I'd encourage you to explore this topic further through research and practice.",
                "That's an interesting question! I'd recommend breaking it down into smaller parts and exploring each aspect systematically.",
                "Great question! Consider looking at this from different perspectives and gathering more information to form a comprehensive understanding.",
                "I appreciate your engagement with learning! Try to connect this concept with what you already know and see how it fits into the bigger picture.",
                "Excellent inquiry! The best way to understand this would be through hands-on practice and seeking additional resources."
            ]
            import random
            return random.choice(fallback_responses)
        
        # Educational tutor prompt
        prompt = f"""You are an intelligent and supportive educational tutor. Help the student learn effectively.

Student's question: {user_input}

Provide a helpful educational response that:
- Addresses their question clearly
- Uses encouraging and supportive language
- Offers practical learning tips
- Suggests next steps for learning

Keep your response concise but informative (2-3 sentences)."""

        # Use text generation API
        response = hf_client.text_generation(
            prompt,
            model="microsoft/DialoGPT-medium",
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )
        
        if response:
            return response
        else:
            return "I understand you're looking for help with learning. Could you provide more specific details about what you'd like to understand better?"
            
    except Exception as e:
        st.warning(f"AI response generation failed: {e}")
        # Enhanced fallback based on input analysis
        if "?" in user_input:
            return f"That's a great question! I'd encourage you to research this topic step by step. Breaking down complex problems into smaller parts often helps with understanding."
        elif any(word in user_input.lower() for word in ['help', 'stuck', 'confused']):
            return "I can see you're working through something challenging. Try approaching it from a different angle, or consider asking for specific examples that might clarify the concept."
        else:
            return "Thank you for sharing that with me. Engaging with learning material actively like this is a great way to deepen your understanding."

# Authentication functions (same as before)
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(db, username, email, password):
    try:
        if not db:
            return False, "Database not available"
            
        users_collection = db['users']
        if users_collection.find_one({"username": username}):
            return False, "Username already exists"
        
        hashed_password = hash_password(password)
        user_data = {
            "username": username,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.now(),
            "analytics": {
                "total_interactions": 0,
                "sessions": [],
                "topics": [],
                "avg_sentiment": 0.5
            }
        }
        users_collection.insert_one(user_data)
        return True, "User registered successfully"
    except Exception as e:
        return False, f"Registration error: {e}"

def authenticate_user(db, username, password):
    try:
        if not db:
            return False, None
            
        users_collection = db['users']
        user = users_collection.find_one({"username": username})
        if user and verify_password(password, user['password']):
            return True, user
        return False, None
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return False, None

# Save interaction to database
def save_interaction(db, username, user_input, ai_response, sentiment_score):
    try:
        if not db:
            return
            
        interactions_collection = db['interactions']
        interaction_data = {
            "username": username,
            "user_input": user_input,
            "ai_response": ai_response,
            "sentiment_score": sentiment_score,
            "timestamp": datetime.now()
        }
        interactions_collection.insert_one(interaction_data)
        
        # Update user analytics
        users_collection = db['users']
        users_collection.update_one(
            {"username": username},
            {
                "$inc": {"analytics.total_interactions": 1},
                "$push": {"analytics.sessions": {
                    "timestamp": datetime.now(),
                    "sentiment": sentiment_score
                }}
            }
        )
    except Exception as e:
        st.error(f"Error saving interaction: {e}")

# Authentication UI (same as before)
def show_auth_page():
    st.markdown('<div class="main-header"><h1>🎓 Edu Tutor AI</h1><p>Intelligent Student Engagement Platform</p></div>', unsafe_allow_html=True)
    
    # Demo mode option
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Try Demo Mode", type="secondary", help="Experience the app without registration"):
            st.session_state.authenticated = True
            st.session_state.username = "demo_user"
            st.rerun()
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary"):
            if username and password:
                db = init_mongodb()
                if db is not None:
                    success, user = authenticate_user(db, username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_data = user
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Database connection failed - using demo mode")
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
            else:
                st.error("Please enter both username and password")
    
    with tab2:
        st.subheader("Create New Account")
        new_username = st.text_input("Username", key="reg_username")
        email = st.text_input("Email", key="reg_email")
        new_password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register", type="primary"):
            if new_username and email and new_password and confirm_password:
                if new_password == confirm_password:
                    db = init_mongodb()
                    if db is not None:
                        success, message = register_user(db, new_username, email, new_password)
                        if success:
                            st.success(message)
                            st.info("Please login with your new credentials")
                        else:
                            st.error(message)
                    else:
                        st.warning("Database connection failed - but you can still use demo mode")
                else:
                    st.error("Passwords do not match")
            else:
                st.error("Please fill in all fields")

# Main Application
def show_main_app():
    db = init_mongodb()
    hf_client = init_ai_client()
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="main-header"><h1>🎓 Edu Tutor AI</h1></div>', unsafe_allow_html=True)
    with col2:
        user_display = st.session_state.username
        if user_display == "demo_user":
            user_display = "Demo User"
        st.write(f"Welcome, **{user_display}**!")
    with col3:
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()
    
    # System status
    with st.expander("🔧 System Status"):
        col1, col2, col3 = st.columns(3)
        with col1:
            db_status = "✅ Connected" if db else "❌ Offline"
            st.write(f"**Database:** {db_status}")
        with col2:
            ai_status = "✅ Connected" if hf_client else "❌ Limited Mode"
            st.write(f"**AI Service:** {ai_status}")
        with col3:
            st.write(f"**Mode:** {'Demo' if st.session_state.username == 'demo_user' else 'Full'}")
    
    # Sidebar with analytics
    with st.sidebar:
        st.header("📊 Dashboard")
        
        # Update session duration
        current_time = datetime.now()
        session_duration = (current_time - st.session_state.user_analytics['start_time']).seconds // 60
        st.session_state.user_analytics['session_duration'] = session_duration
        
        # Metrics
        st.metric("Session Duration", f"{session_duration} minutes")
        st.metric("Interactions", st.session_state.user_analytics['total_interactions'])
        
        # Sentiment gauge
        avg_sentiment = st.session_state.user_analytics['avg_sentiment']
        st.subheader("Current Mood")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_sentiment * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ],
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Main chat interface
    st.header("💬 AI Tutor Chat")
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for i, (role, message, sentiment) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message ai-message"><strong>AI Tutor:</strong> {message}<br><small>Detected mood: {sentiment}</small></div>', unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Ask your question or share your thoughts:", key="chat_input", placeholder="Type your message here...")
    with col2:
        send_button = st.button("Send", type="primary")
    
    # Process user input
    if send_button and user_input:
        # Analyze sentiment
        sentiment_score, sentiment_label = analyze_sentiment_api(hf_client, user_input)
        
        # Generate AI response
        with st.spinner("Thinking..."):
            context = f"Previous interactions: {len(st.session_state.chat_history)}"
            ai_response = generate_ai_response(hf_client, user_input, context)
        
        # Update chat history
        st.session_state.chat_history.append(("user", user_input, ""))
        st.session_state.chat_history.append(("assistant", ai_response, sentiment_label))
        
        # Update analytics
        st.session_state.user_analytics['total_interactions'] += 1
        
        # Update average sentiment
        current_avg = st.session_state.user_analytics['avg_sentiment']
        total_interactions = st.session_state.user_analytics['total_interactions']
        new_avg = ((current_avg * (total_interactions - 1)) + sentiment_score) / total_interactions
        st.session_state.user_analytics['avg_sentiment'] = new_avg
        
        # Save to database (only if connected)
        if db and st.session_state.username != "demo_user":
            save_interaction(db, st.session_state.username, user_input, ai_response, sentiment_score)
        
        # Clear input and refresh
        st.rerun()
    
    # Analytics Dashboard
    st.header("📈 Learning Analytics")
    
    if st.session_state.chat_history:
        # Extract sentiment data for visualization
        sentiments = []
        timestamps = []
        for i, (role, message, sentiment) in enumerate(st.session_state.chat_history):
            if role == "user":
                # Extract sentiment score from label
                if "Positive" in str(sentiment):
                    score = 70
                elif "Negative" in str(sentiment):
                    score = 30
                else:
                    score = 50
                sentiments.append(score)
                timestamps.append(f"Interaction {len(timestamps) + 1}")
        
        if sentiments:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment trend line chart
                fig_trend = px.line(
                    x=timestamps, 
                    y=sentiments,
                    title="Sentiment Trend Throughout Session",
                    labels={"x": "Interactions", "y": "Sentiment Score (%)"}
                )
                fig_trend.update_traces(line_color='#667eea')
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Sentiment distribution
                sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
                for _, (role, message, sentiment) in enumerate(st.session_state.chat_history):
                    if role == "user" and sentiment:
                        if "Positive" in str(sentiment):
                            sentiment_counts["Positive"] += 1
                        elif "Negative" in str(sentiment):
                            sentiment_counts["Negative"] += 1
                        else:
                            sentiment_counts["Neutral"] += 1
                
                fig_pie = px.pie(
                    values=list(sentiment_counts.values()),
                    names=list(sentiment_counts.keys()),
                    title="Overall Sentiment Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Start chatting to see your learning analytics! 🚀")

# Main app logic
def main():
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
