
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
from transformers import pipeline
import os
from huggingface_hub import InferenceClient
import time

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
        background-color: black;
        margin-left: 2rem;
    }
    
    .ai-message {
        background-color: black;
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
    MONGODB_URI = st.secrets.get("MONGODB_URI", "")  #mongodb+srv://your-connection-string in secrets you will give
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")  #your-huggingface-token in secrets
    # IBM_API_KEY = st.secrets.get("IBM_API_KEY", "your-ibm-api-key")
    # IBM_PROJECT_ID = st.secrets.get("IBM_PROJECT_ID", "your-project-id")

# MongoDB connection
@st.cache_resource
def init_mongodb():
    try:
        client = MongoClient(Config.MONGODB_URI)
        db = client['edu_tutor_ai']
        return db
    except Exception as e:
        st.error(f"MongoDB connection error: {e}")
        return None

# Initialize AI models
@st.cache_resource
def init_ai_models():
    try:
        # Initialize sentiment analyzer with fallback
        try:
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except:
            # Fallback to a lighter model
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
        
        # Hugging Face client for Granite model
        hf_client = None
        if Config.HF_TOKEN and Config.HF_TOKEN != "your-huggingface-token":
            try:
                hf_client = InferenceClient(
                    model="ibm-granite/granite-3.3-2b-instruct",
                    token=Config.HF_TOKEN
                )
            except:
                # Fallback to a different model
                hf_client = InferenceClient(
                    model="microsoft/DialoGPT-medium",
                    token=Config.HF_TOKEN
                )
                #st.write("HF Token loaded:", bool(Config.HF_TOKEN and Config.HF_TOKEN.strip()))

        
        return sentiment_analyzer, hf_client
    except Exception as e:
        st.warning(f"Some AI features may be limited: {e}")
        return None, None

# Authentication functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(db, username, email, password):
    try:
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
        users_collection = db['users']
        user = users_collection.find_one({"username": username})
        if user and verify_password(password, user['password']):
            return True, user
        return False, None
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return False, None

# AI Response Generation
def generate_ai_response(hf_client, user_input, context=""):
    try:
        if not hf_client:
            return "❌ AI model not initialized. Please check your Hugging Face token in Streamlit secrets."
        
        prompt = f"""You are an intelligent educational tutor AI assistant. Your role is to help students learn effectively by providing clear, engaging, and personalized responses.

Context: {context}
Student Question: {user_input}

Please provide a helpful, educational response that:
1. Addresses the student's question directly
2. Uses simple, clear language appropriate for learning
3. Provides examples when helpful
4. Encourages further learning
5. Is supportive and motivating

Response:"""

        response = hf_client.text_generation(
            prompt,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )
        return response

    except Exception as e:
        return f"❌ AI model error: {str(e)}. Please verify your Hugging Face token and model settings."

#         import random
#         return random.choice(fallback_responses)


# Sentiment Analysis
def analyze_sentiment(sentiment_analyzer, text):
    try:
        results = sentiment_analyzer(text)
        # Convert to simple positive/negative/neutral score
        sentiment_score = 0
        for result in results[0]:
            if result['label'] == 'LABEL_2':  # Positive
                sentiment_score += result['score']
            elif result['label'] == 'LABEL_0':  # Negative
                sentiment_score -= result['score']
        
        # Normalize to 0-1 scale
        sentiment_score = (sentiment_score + 1) / 2
        
        if sentiment_score > 0.6:
            sentiment_label = "😊 Positive"
        elif sentiment_score < 0.4:
            sentiment_label = "😔 Negative"
        else:
            sentiment_label = "😐 Neutral"
            
        return sentiment_score, sentiment_label
    except Exception as e:
        return 0.5, "😐 Neutral"

# Save interaction to database
def save_interaction(db, username, user_input, ai_response, sentiment_score):
    try:
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

# Authentication UI
def show_auth_page():
    st.markdown('<div class="main-header"><h1>🎓 Edu Tutor AI</h1><p>Intelligent Student Engagement Platform</p></div>', unsafe_allow_html=True)
    
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
                    st.error("Database connection failed")
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
                        st.error("Database connection failed")
                else:
                    st.error("Passwords do not match")
            else:
                st.error("Please fill in all fields")

# Main Application
def show_main_app():
    db = init_mongodb()
    sentiment_analyzer, hf_client = init_ai_models()
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="main-header"><h1>🎓 Edu Tutor AI</h1></div>', unsafe_allow_html=True)
    with col2:
        st.write(f"Welcome, **{st.session_state.username}**!")
    with col3:
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()
    
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
            mode="gauge+number+delta",
            value=avg_sentiment * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 60], 'color': "gray"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
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
        if hf_client and sentiment_analyzer:
            # Analyze sentiment
            sentiment_score, sentiment_label = analyze_sentiment(sentiment_analyzer, user_input)
            
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
            
            # Save to database
            if db is not None:
                save_interaction(db, st.session_state.username, user_input, ai_response, sentiment_score)
            
            # Clear input and refresh
            st.rerun()
    
    # Analytics Dashboard
    st.header("📈 Learning Analytics")
    
    if st.session_state.chat_history:
        # Sentiment trend
        sentiments = []
        timestamps = []
        for i, (role, message, sentiment) in enumerate(st.session_state.chat_history):
            if role == "user":
                # Extract sentiment score from label (simplified)
                if "Positive" in str(sentiment):
                    score = 0.8
                elif "Negative" in str(sentiment):
                    score = 0.2
                else:
                    score = 0.5
                sentiments.append(score * 100)
                timestamps.append(i)
        
        if sentiments:
            fig_trend = px.line(
                x=timestamps, 
                y=sentiments,
                title="Sentiment Trend Throughout Session",
                labels={"x": "Interaction Number", "y": "Sentiment Score (%)"}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Start chatting to see your learning analytics!")

# Main app logic
def main():
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        show_main_app()

if __name__ == "__main__":
    main()



# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime, timedelta
# import pymongo
# from pymongo import MongoClient
# import bcrypt
# import requests
# import json
# from transformers import pipeline
# import os
# from huggingface_hub import InferenceClient
# import time

# # Page configuration
# st.set_page_config(
#     page_title="Edu Tutor AI - Intelligent Student Engagement Platform",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better UI
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
    
#     .metric-card {
#         background: white;
#         padding: 1rem;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         border-left: 4px solid #667eea;
#     }
    
#     .chat-message {
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 0.5rem 0;
#         border-left: 4px solid #667eea;
#     }
    
#     .user-message {
#         background-color: #e3f2fd;
#         margin-left: 2rem;
#     }
    
#     .ai-message {
#         background-color: #f3e5f5;
#         margin-right: 2rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'authenticated' not in st.session_state:
#     st.session_state.authenticated = False
# if 'username' not in st.session_state:
#     st.session_state.username = None
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'user_analytics' not in st.session_state:
#     st.session_state.user_analytics = {
#         'total_interactions': 0,
#         'avg_sentiment': 0.5,
#         'topics_discussed': [],
#         'session_duration': 0,
#         'start_time': datetime.now()
#     }

# # Configuration
# class Config:
#     MONGODB_URI = st.secrets.get("MONGODB_URI", " ") #mongodb+srv://your-connection-string
#     HF_TOKEN = st.secrets.get("HF_TOKEN", " ")  #your-huggingface-token
#     # IBM_API_KEY = st.secrets.get("IBM_API_KEY", "your-ibm-api-key")
#     # IBM_PROJECT_ID = st.secrets.get("IBM_PROJECT_ID", "your-project-id")

# # MongoDB connection
# @st.cache_resource
# def init_mongodb():
#     try:
#         client = MongoClient(Config.MONGODB_URI)
#         db = client['edu_tutor_ai']
#         return db
#     except Exception as e:
#         st.error(f"MongoDB connection error: {e}")
#         return None

# # Initialize AI models
# @st.cache_resource
# def init_ai_models():
#     try:
#         # Sentiment analysis pipeline
#         sentiment_analyzer = pipeline(
#             "sentiment-analysis",
#             model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#             return_all_scores=True
#         )
        
#         # Hugging Face client for Granite model
#         hf_client = InferenceClient(
#             model="ibm-granite/granite-3b-code-instruct",
#             token=Config.HF_TOKEN
#         )
        
#         return sentiment_analyzer, hf_client
#     except Exception as e:
#         st.error(f"Error initializing AI models: {e}")
#         return None, None

# # Authentication functions
# def hash_password(password):
#     return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# def verify_password(password, hashed):
#     return bcrypt.checkpw(password.encode('utf-8'), hashed)

# def register_user(db, username, email, password):
#     try:
#         users_collection = db['users']
#         if users_collection.find_one({"username": username}):
#             return False, "Username already exists"
        
#         hashed_password = hash_password(password)
#         user_data = {
#             "username": username,
#             "email": email,
#             "password": hashed_password,
#             "created_at": datetime.now(),
#             "analytics": {
#                 "total_interactions": 0,
#                 "sessions": [],
#                 "topics": [],
#                 "avg_sentiment": 0.5
#             }
#         }
#         users_collection.insert_one(user_data)
#         return True, "User registered successfully"
#     except Exception as e:
#         return False, f"Registration error: {e}"

# def authenticate_user(db, username, password):
#     try:
#         users_collection = db['users']
#         user = users_collection.find_one({"username": username})
#         if user and verify_password(password, user['password']):
#             return True, user
#         return False, None
#     except Exception as e:
#         st.error(f"Authentication error: {e}")
#         return False, None

# # AI Response Generation
# def generate_ai_response(hf_client, user_input, context=""):
#     try:
#         prompt = f"""You are an intelligent educational tutor AI assistant. Your role is to help students learn effectively by providing clear, engaging, and personalized responses.

# Context: {context}
# Student Question: {user_input}

# Please provide a helpful, educational response that:
# 1. Addresses the student's question directly
# 2. Uses simple, clear language appropriate for learning
# 3. Provides examples when helpful
# 4. Encourages further learning
# 5. Is supportive and motivating

# Response:"""

#         response = hf_client.text_generation(
#             prompt,
#             max_new_tokens=500,
#             temperature=0.7,
#             do_sample=True
#         )
#         return response
#     except Exception as e:
#         return f"I apologize, but I'm having trouble generating a response right now. Error: {e}"

# # Sentiment Analysis
# def analyze_sentiment(sentiment_analyzer, text):
#     try:
#         results = sentiment_analyzer(text)
#         # Convert to simple positive/negative/neutral score
#         sentiment_score = 0
#         for result in results[0]:
#             if result['label'] == 'LABEL_2':  # Positive
#                 sentiment_score += result['score']
#             elif result['label'] == 'LABEL_0':  # Negative
#                 sentiment_score -= result['score']
        
#         # Normalize to 0-1 scale
#         sentiment_score = (sentiment_score + 1) / 2
        
#         if sentiment_score > 0.6:
#             sentiment_label = "😊 Positive"
#         elif sentiment_score < 0.4:
#             sentiment_label = "😔 Negative"
#         else:
#             sentiment_label = "😐 Neutral"
            
#         return sentiment_score, sentiment_label
#     except Exception as e:
#         return 0.5, "😐 Neutral"

# # Save interaction to database
# def save_interaction(db, username, user_input, ai_response, sentiment_score):
#     try:
#         interactions_collection = db['interactions']
#         interaction_data = {
#             "username": username,
#             "user_input": user_input,
#             "ai_response": ai_response,
#             "sentiment_score": sentiment_score,
#             "timestamp": datetime.now()
#         }
#         interactions_collection.insert_one(interaction_data)
        
#         # Update user analytics
#         users_collection = db['users']
#         users_collection.update_one(
#             {"username": username},
#             {
#                 "$inc": {"analytics.total_interactions": 1},
#                 "$push": {"analytics.sessions": {
#                     "timestamp": datetime.now(),
#                     "sentiment": sentiment_score
#                 }}
#             }
#         )
#     except Exception as e:
#         st.error(f"Error saving interaction: {e}")

# # Authentication UI
# def show_auth_page():
#     st.markdown('<div class="main-header"><h1>🎓 Edu Tutor AI</h1><p>Intelligent Student Engagement Platform</p></div>', unsafe_allow_html=True)
    
#     tab1, tab2 = st.tabs(["Login", "Register"])
    
#     with tab1:
#         st.subheader("Login to Your Account")
#         username = st.text_input("Username", key="login_username")
#         password = st.text_input("Password", type="password", key="login_password")
        
#         if st.button("Login", type="primary"):
#             if username and password:
#                 db = init_mongodb()
#                 if db is not None:
#                     success, user = authenticate_user(db, username, password)
#                     if success:
#                         st.session_state.authenticated = True
#                         st.session_state.username = username
#                         st.session_state.user_data = user
#                         st.rerun()
#                     else:
#                         st.error("Invalid username or password")
#                 else:
#                     st.error("Database connection failed")
#             else:
#                 st.error("Please enter both username and password")
    
#     with tab2:
#         st.subheader("Create New Account")
#         new_username = st.text_input("Username", key="reg_username")
#         email = st.text_input("Email", key="reg_email")
#         new_password = st.text_input("Password", type="password", key="reg_password")
#         confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
#         if st.button("Register", type="primary"):
#             if new_username and email and new_password and confirm_password:
#                 if new_password == confirm_password:
#                     db = init_mongodb()
#                     if db is not None:
#                         success, message = register_user(db, new_username, email, new_password)
#                         if success:
#                             st.success(message)
#                             st.info("Please login with your new credentials")
#                         else:
#                             st.error(message)
#                     else:
#                         st.error("Database connection failed")
#                 else:
#                     st.error("Passwords do not match")
#             else:
#                 st.error("Please fill in all fields")

# # Main Application
# def show_main_app():
#     db = init_mongodb()
#     sentiment_analyzer, hf_client = init_ai_models()
    
#     # Header
#     col1, col2, col3 = st.columns([2, 1, 1])
#     with col1:
#         st.markdown('<div class="main-header"><h1>🎓 Edu Tutor AI</h1></div>', unsafe_allow_html=True)
#     with col2:
#         st.write(f"Welcome, **{st.session_state.username}**!")
#     with col3:
#         if st.button("Logout", type="secondary"):
#             st.session_state.authenticated = False
#             st.session_state.username = None
#             st.session_state.chat_history = []
#             st.rerun()
    
#     # Sidebar with analytics
#     with st.sidebar:
#         st.header("📊 Dashboard")
        
#         # Update session duration
#         current_time = datetime.now()
#         session_duration = (current_time - st.session_state.user_analytics['start_time']).seconds // 60
#         st.session_state.user_analytics['session_duration'] = session_duration
        
#         # Metrics
#         st.metric("Session Duration", f"{session_duration} minutes")
#         st.metric("Interactions", st.session_state.user_analytics['total_interactions'])
        
#         # Sentiment gauge
#         avg_sentiment = st.session_state.user_analytics['avg_sentiment']
#         st.subheader("Current Mood")
        
#         fig_gauge = go.Figure(go.Indicator(
#             mode="gauge+number+delta",
#             value=avg_sentiment * 100,
#             domain={'x': [0, 1], 'y': [0, 1]},
#             title={'text': "Sentiment Score"},
#             gauge={
#                 'axis': {'range': [None, 100]},
#                 'bar': {'color': "lightblue"},
#                 'steps': [
#                     {'range': [0, 40], 'color': "lightgray"},
#                     {'range': [40, 60], 'color': "gray"},
#                     {'range': [60, 100], 'color': "lightgreen"}
#                 ],
#                 'threshold': {
#                     'line': {'color': "red", 'width': 4},
#                     'thickness': 0.75,
#                     'value': 90
#                 }
#             }
#         ))
#         fig_gauge.update_layout(height=300)
#         st.plotly_chart(fig_gauge, use_container_width=True)
    
#     # Main chat interface
#     st.header("💬 AI Tutor Chat")
    
#     # Chat history display
#     chat_container = st.container()
#     with chat_container:
#         for i, (role, message, sentiment) in enumerate(st.session_state.chat_history):
#             if role == "user":
#                 st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message}</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="chat-message ai-message"><strong>AI Tutor:</strong> {message}<br><small>Detected mood: {sentiment}</small></div>', unsafe_allow_html=True)
    
#     # Chat input
#     col1, col2 = st.columns([4, 1])
#     with col1:
#         user_input = st.text_input("Ask your question or share your thoughts:", key="chat_input", placeholder="Type your message here...")
#     with col2:
#         send_button = st.button("Send", type="primary")
    
#     # Process user input
#     if (send_button or user_input) and user_input:
#         if hf_client and sentiment_analyzer:
#             # Analyze sentiment
#             sentiment_score, sentiment_label = analyze_sentiment(sentiment_analyzer, user_input)
            
#             # Generate AI response
#             with st.spinner("Thinking..."):
#                 context = f"Previous interactions: {len(st.session_state.chat_history)}"
#                 ai_response = generate_ai_response(hf_client, user_input, context)
            
#             # Update chat history
#             st.session_state.chat_history.append(("user", user_input, ""))
#             st.session_state.chat_history.append(("assistant", ai_response, sentiment_label))
            
#             # Update analytics
#             st.session_state.user_analytics['total_interactions'] += 1
            
#             # Update average sentiment
#             current_avg = st.session_state.user_analytics['avg_sentiment']
#             total_interactions = st.session_state.user_analytics['total_interactions']
#             new_avg = ((current_avg * (total_interactions - 1)) + sentiment_score) / total_interactions
#             st.session_state.user_analytics['avg_sentiment'] = new_avg
            
#             # Save to database
#             if db:
#                 save_interaction(db, st.session_state.username, user_input, ai_response, sentiment_score)
            
#             # Clear input and refresh
#             st.rerun()
    
#     # Analytics Dashboard
#     st.header("📈 Learning Analytics")
    
#     if st.session_state.chat_history:
#         # Sentiment trend
#         sentiments = []
#         timestamps = []
#         for i, (role, message, sentiment) in enumerate(st.session_state.chat_history):
#             if role == "user":
#                 # Extract sentiment score from label (simplified)
#                 if "Positive" in str(sentiment):
#                     score = 0.8
#                 elif "Negative" in str(sentiment):
#                     score = 0.2
#                 else:
#                     score = 0.5
#                 sentiments.append(score * 100)
#                 timestamps.append(i)
        
#         if sentiments:
#             fig_trend = px.line(
#                 x=timestamps, 
#                 y=sentiments,
#                 title="Sentiment Trend Throughout Session",
#                 labels={"x": "Interaction Number", "y": "Sentiment Score (%)"}
#             )
#             st.plotly_chart(fig_trend, use_container_width=True)
#     else:
#         st.info("Start chatting to see your learning analytics!")

# # Main app logic
# def main():
#     if not st.session_state.authenticated:
#         show_auth_page()
#     else:
#         show_main_app()

# if __name__ == "__main__":
#     main()




