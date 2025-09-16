
import streamlit as st
import requests
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

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

HF_BASE = "https://api-inference.huggingface.co/models"

def call_hf(model_id, inputs):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    resp = requests.post(f"{HF_BASE}/{model_id}", headers=headers, json={"inputs": inputs})
    return resp.json()

def parse_text(resp):
    if isinstance(resp, list) and "generated_text" in resp[0]:
        return resp[0]["generated_text"]
    return str(resp)

def get_reply(prompt, context=""):
    model_id = "ibm-granite/granite-3.3-2b-instruct"
    payload = f"Context: {context}\nStudent: {prompt}"
    return parse_text(call_hf(model_id, payload))

def sentiment(text):
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    resp = call_hf(model_id, text)
    if isinstance(resp, list) and len(resp) > 0:
        return resp[0]["label"], resp[0]["score"]
    return "UNKNOWN", 0.0

def signup(email, pwd):
    if users_col.find_one({"email": email}): return False
    users_col.insert_one({"email": email, "pwd": pwd})
    return True

def login(email, pwd):
    return users_col.find_one({"email": email, "pwd": pwd})

def save_msg(email, role, text, sent):
    chats_col.insert_one({"email": email, "role": role, "text": text,
                          "sentiment": sent, "ts": datetime.utcnow()})

def load_chats(email):
    return list(chats_col.find({"email": email}).sort("ts", 1))

st.title("Edu Tutor AI")

if "user" not in st.session_state: st.session_state.user = None

with st.sidebar:
    st.header("Account")
    if not st.session_state.user:
        mode = st.radio("Mode", ["Login", "Signup"])
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        if st.button("Submit"):
            if mode == "Signup":
                ok = signup(email, pwd)
                st.info("Signup ok" if ok else "User exists")
            else:
                u = login(email, pwd)
                if u:
                    st.session_state.user = u
                    st.experimental_rerun()
                else:
                    st.error("Wrong login")
    else:
        st.success(f"Logged in as {st.session_state.user['email']}")
        if st.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()

if st.session_state.user:
    email = st.session_state.user["email"]
    st.subheader("Chat")
    for c in load_chats(email):
        st.write(f"{c['role']}: {c['text']} ({c['sentiment']})")
    msg = st.text_area("Your question")
    if st.button("Ask"):
        label, score = sentiment(msg)
        save_msg(email, "user", msg, (label, score))
        reply = get_reply(msg)
        r_label, r_score = sentiment(reply)
        save_msg(email, "assistant", reply, (r_label, r_score))
        st.experimental_rerun()

    st.subheader("Dashboard")
    total_users = users_col.count_documents({})
    total_msgs = chats_col.count_documents({})
    st.metric("Users", total_users)
    st.metric("Messages", total_msgs)
