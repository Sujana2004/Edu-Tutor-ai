
import streamlit as st
import requests
from pymongo import MongoClient
from datetime import datetime
from passlib.hash import bcrypt

st.set_page_config(page_title="Edu Tutor AI", layout="wide")

HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", "")
MONGO_URI = st.secrets.get("MONGO_URI", "")

if not HF_TOKEN or not MONGO_URI:
    st.warning("Please set HUGGINGFACE_TOKEN and MONGO_URI in Streamlit secrets.")
    st.stop()

client = MongoClient(MONGO_URI)
db = client.get_database("edu_tutor_ai")
users_col = db.get_collection("users")
chats_col = db.get_collection("chats")

HF_BASE = "https://huggingface.co/Sujana85/models"

def call_hf(model_id, inputs):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        resp = requests.post(f"{HF_BASE}/{model_id}", headers=headers, json={"inputs": inputs}, timeout=60)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"error": "Invalid JSON response from Hugging Face API"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def parse_text(resp):
    if isinstance(resp, dict) and "error" in resp:
        return f"Error: {resp['error']}"
    if isinstance(resp, list) and "generated_text" in resp[0]:
        return resp[0]["generated_text"]
    return str(resp)

def get_reply(prompt, context=""):
    model_id = "granite-3.3-2b-instruct"
    payload = f"Context: {context}\nStudent: {prompt}"
    return parse_text(call_hf(model_id, payload))

def sentiment(text):
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    resp = call_hf(model_id, text)
    if isinstance(resp, list) and len(resp) > 0 and "label" in resp[0] and "score" in resp[0]:
        return resp[0]["label"], resp[0]["score"]
    return "UNKNOWN", 0.0

def signup(email, pwd):
    if users_col.find_one({"email": email}):
        return False
    hashed = bcrypt.hash(pwd)
    users_col.insert_one({"email": email, "pwd": hashed})
    return True

def login(email, pwd):
    user = users_col.find_one({"email": email})
    if user and bcrypt.verify(pwd, user["pwd"]):
        return user
    return None

def save_msg(email, role, text, sent):
    chats_col.insert_one({
        "email": email,
        "role": role,
        "text": text,
        "sentiment": sent,
        "ts": datetime.utcnow()
    })

def load_chats(email):
    return list(chats_col.find({"email": email}).sort("ts", 1))

st.title("Edu Tutor AI")

if "user" not in st.session_state:
    st.session_state.user = None
if "last_reply" not in st.session_state:
    st.session_state.last_reply = ""

with st.sidebar:
    st.header("Account")
    if not st.session_state.user:
        mode = st.radio("Mode", ["Login", "Signup"])
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        if st.button("Submit"):
            if mode == "Signup":
                ok = signup(email, pwd)
                st.info("Signup successful" if ok else "User already exists")
            else:
                u = login(email, pwd)
                if u:
                    st.session_state.user = u
                else:
                    st.error("Wrong email or password")
    else:
        st.success(f"Logged in as {st.session_state.user['email']}")
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.last_reply = ""

if st.session_state.user:
    email = st.session_state.user["email"]

    st.subheader("Chat History")
    for c in load_chats(email):
        st.write(f"{c['role']}: {c['text']} (Sentiment: {c['sentiment']})")

    st.subheader("Ask a Question")
    msg = st.text_area("Your question")

    if st.button("Ask"):
        if msg.strip():
            label, score = sentiment(msg)
            save_msg(email, "user", msg, (label, score))

            reply = get_reply(msg)
            r_label, r_score = sentiment(reply)
            save_msg(email, "assistant", reply, (r_label, r_score))

            st.session_state.last_reply = reply
        else:
            st.warning("Please enter a valid question.")

    if st.session_state.last_reply:
        st.write(f"**Assistant:** {st.session_state.last_reply}")

    st.subheader("Dashboard")
    total_users = users_col.count_documents({})
    total_msgs = chats_col.count_documents({})
    st.metric("Users", total_users)
    st.metric("Messages", total_msgs)

