import streamlit as st
import requests
import uuid

# ================= CONFIG =================
API_URL = "http://localhost:8000/chat"   # change if deployed
APP_TITLE = "🧠 AI Research & Persona Agent"

# ================= SESSION =================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{uuid.uuid4().hex[:12]}"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= PAGE =================
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title(APP_TITLE)

# ================= CHAT DISPLAY =================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= INPUT =================
user_input = st.chat_input("Ask something... (persona, research, email, etc.)")

if user_input:
    # Show user message immediately
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend
    try:
        response = requests.post(
            API_URL,
            headers={
                "session-id": st.session_state.session_id
            },
            json={
                "message": user_input
            },
            timeout=60
        )

        data = response.json()

        # Extract last AI message
        if "history" in data and data["history"]:
            ai_reply = data["history"][-1]["ai"]
        else:
            ai_reply = "No response received."

    except Exception as e:
        ai_reply = f"❌ Error: {e}"

    # Show AI reply
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": ai_reply
    })

    with st.chat_message("assistant"):
        st.markdown(ai_reply)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Controls")

    st.write("Session ID:")
    st.code(st.session_state.session_id)

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


    st.markdown("---")
 