# 🎓 Edu Tutor AI - Intelligent Student Engagement Platform

A comprehensive AI-powered educational platform built with Streamlit, featuring real-time conversational AI, sentiment analysis, dynamic dashboards, and personalized learning experiences.

## 🌟 Features

### 1. **Real-Time Conversational AI Assistant**
- Powered by IBM Granite model via Hugging Face
- Educational context-aware responses
- Natural language understanding
- Supportive and motivating interactions

### 2. **Sentiment Analysis**
- Real-time mood detection from student inputs
- Emotional engagement tracking
- Adaptive responses based on student sentiment
- Visual sentiment indicators

### 3. **Dynamic Dashboard**
- Real-time analytics and metrics
- Session duration tracking
- Interaction count monitoring
- Sentiment trend visualization
- Interactive charts and gauges

### 4. **Personalized & Contextual Response System**
- User authentication and profiles
- MongoDB Atlas integration for data persistence
- Personalized learning paths
- Context-aware conversations
- Session history tracking

## 🚀 Quick Start

### Prerequisites

1. **Accounts needed:**
   - GitHub account
   - Hugging Face account
   - MongoDB Atlas account
   - Streamlit Cloud account (optional for deployment)

### Setup Steps

#### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/edu-tutor-ai.git
cd edu-tutor-ai

# Install dependencies locally (optional)
pip install -r requirements.txt
```

#### 2. MongoDB Atlas Setup

1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a new cluster (free tier available)
3. Create a database user
4. Get your connection string
5. Whitelist your IP address or use 0.0.0.0/0 for all IPs

#### 3. Hugging Face Setup

1. Go to [Hugging Face](https://huggingface.co/)
2. Create an account and get your API token
3. Go to Settings > Access Tokens
4. Create a new token with read permissions

#### 4. IBM Granite Model Access

**Option 1: Via Hugging Face (Recommended)**
- The app uses `ibm-granite/granite-3b-code-instruct` model
- No additional setup required beyond HF token

**Option 2: Via IBM watsonx.ai (Advanced)**
1. Sign up for IBM Cloud
2. Access watsonx.ai platform
3. Get API key and project ID
4. Update the app code to use IBM SDK

#### 5. Configuration

Create `.streamlit/secrets.toml` file:

```toml
# MongoDB Atlas connection
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/edu_tutor_ai?retryWrites=true&w=majority"

# Hugging Face API Token
HF_TOKEN = "hf_your_token_here"

# IBM credentials (optional)
IBM_API_KEY = "your_ibm_api_key"
IBM_PROJECT_ID = "your_project_id"
```

#### 6. Local Development

```bash
# Run the app locally
streamlit run app.py
```

#### 7. GitHub Deployment

1. **Push to GitHub:**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **File Structure:**
```
edu-tutor-ai/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.toml
└── .gitignore
```

#### 8. Streamlit Cloud Deployment

1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Connect your GitHub account
3. Select your repository
4. Set main file as `app.py`
5. Add secrets in the Streamlit Cloud dashboard:
   - Go to App Settings > Secrets
   - Copy content from your `.streamlit/secrets.toml`

## 🔧 Configuration Details

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB Atlas connection string | Yes |
| `HF_TOKEN` | Hugging Face API token | Yes |
| `IBM_API_KEY` | IBM Cloud API key | Optional |
| `IBM_PROJECT_ID` | IBM watsonx project ID | Optional |

### Database Schema

**Users Collection:**
```json
{
  "username": "string",
  "email": "string", 
  "password": "hashed_string",
  "created_at": "datetime",
  "analytics": {
    "total_interactions": "number",
    "sessions": "array",
    "topics": "array",
    "avg_sentiment": "number"
  }
}
```

**Interactions Collection:**
```json
{
  "username": "string",
  "user_input": "string",
  "ai_response": "string", 
  "sentiment_score": "number",
  "timestamp": "datetime"
}
```

## 🎯 Usage

### For Students:
1. Register/Login to your account
2. Start chatting with the AI tutor
3. Ask questions, share thoughts, or discuss topics
4. View your learning analytics and mood trends
5. Track your engagement and progress

### For Educators:
- Monitor student engagement through analytics
- Understand student sentiment and emotional state
- Track learning patterns and interactions
- Provide personalized support based on data

## 🛠️ Technical Architecture

### Frontend:
- **Streamlit** - Web application framework
- **Plotly** - Interactive charts and visualizations
- **Custom CSS** - Enhanced UI/UX

### Backend:
- **MongoDB Atlas** - User data and analytics storage
- **Hugging Face Hub** - AI model inference
- **IBM Granite** - Large language model
- **Transformers** - Sentiment analysis

### AI Models:
- **IBM Granite 3B Code Instruct** - Main conversational AI
- **Cardiff Twitter RoBERTa** - Sentiment analysis
- **Hugging Face Inference API** - Model serving

## 🔒 Security Features

- Password hashing with bcrypt
- User session management
- Secure database connections
- API token protection
- Input validation and sanitization

## 📊 Analytics Features

### Real-time Metrics:
- Session duration tracking
- Interaction count monitoring
- Sentiment score calculation
- Mood trend visualization

### Dashboard Components:
- Sentiment gauge (0-100 scale)
- Interaction timeline
- Mood trend charts
- Session statistics

## 🚀 Deployment Options

### 1. Streamlit Cloud (Recommended)
- Free hosting for public repos
- Automatic deployments from GitHub
- Built-in secrets management
- Custom domains available

### 2. Local Development
- Full control over environment
- Faster iteration during development
- Local database testing

### 3. Other Platforms
- Heroku
- Railway
- DigitalOcean App Platform
- AWS/GCP/Azure

## 🔄 Future Enhancements

- [ ] Multi-language support
- [ ] Voice interaction capabilities  
- [ ] Advanced learning analytics
- [ ] Integration with LMS platforms
- [ ] Mobile app development
- [ ] Offline mode capabilities
- [ ] Advanced AI model fine-tuning
- [ ] Real-time collaboration features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Support

For support and questions:
- Create an issue on GitHub
- Contact: your-email@example.com

## 🙏 Acknowledgments

- IBM for Granite model access
- Hugging Face for model hosting
- MongoDB for database services
- Streamlit for the amazing framework
- Cardiff NLP for sentiment analysis model

---

**Happy Learning! 🎓✨**
