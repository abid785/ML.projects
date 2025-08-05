import streamlit as st
from openai import AsyncOpenAI
import datetime
import json
import os
import asyncio

# === Configuration ===
MODELS = [
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-opus",
    "mistralai/mixtral-8x7b-instruct",
    "openchat/openchat-3.5"
]

# === Setup ===
def init_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_tokens" not in st.session_state:
        st.session_state.last_tokens = 0

def setup_page():
    st.set_page_config(page_title="🧠🖋️ WRITER", layout="wide")
    st.markdown("""
        <style>
            .stApp {
                background-color: white;
                color: black;
            }
            /* === TOP BAR === */
            .top-bar {
                background-color: #2c2c2c;
                height: 40px;
                width: 100%;
                position: fixed;
                top: 0;
                left: 0;
                z-index: 999;
            }
            .main-heading {
                margin-top: 60px; /* push heading down due to bar */
            }
            .message-user {
                background-color: #e7f3ff;
                border-radius: 15px 15px 0 15px;
                padding: 12px;
                margin: 10px 0 10px 20%;
                color: black;
            }
            .message-bot {
                background-color: #f3e7ff;
                border-radius: 15px 15px 15px 0;
                padding: 12px;
                margin: 10px 20% 10px 0;
                color: black;
            }
            .token-counter {
                font-size: 12px;
                color: #555;
                text-align: right;
                margin: -10px 0 10px 0;
            }
            .stSidebar {
                background-color: #f5f5f5 !important;
                color: black;
            }
        </style>
        <div class="top-bar"></div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-heading' style='text-align: center; color: black;'>🧠🖋️ WRITER</h1>", unsafe_allow_html=True)

# === Chat Functions ===
async def get_ai_response(client, messages, model):
    try:
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def save_chat(history, model, token_count):
    if not history: return None
    os.makedirs("chats", exist_ok=True)
    chat_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"chats/chat_{chat_id}.json", "w") as f:
        json.dump({
            "metadata": {"created_at": chat_id, "model": model, "token_count": token_count},
            "messages": history
        }, f, indent=2)
    return chat_id

# === UI Components ===
def sidebar():
    with st.sidebar:
        st.title("📚 MENU")
        option = st.radio("Navigate", [
            "💬 Current Chat", "🌟 New Chat",
            "📖 Chat Library", "⚙️ Settings"
        ], label_visibility="collapsed")
        st.divider()
        model = st.selectbox("🤖 Model", MODELS, index=0)
        st.divider()
        if st.button("🧹 Clear Current Chat"):
            st.session_state.chat_history = []
            st.rerun()
        return option, model

def display_chat(history):
    for sender, msg in history:
        css_class = "message-user" if sender == "🧑 You" else "message-bot"
        st.markdown(f'<div class="{css_class}"><b>{sender}:</b> {msg}</div>',
                    unsafe_allow_html=True)

# === Main App ===
async def main():
    setup_page()
    init_session()
    option, model = sidebar()

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"  # Replace with your valid OpenRouter API key
    )

    if option in ["💬 Current Chat", "🌟 New Chat"]:
        display_chat(st.session_state.chat_history)

        if prompt := st.chat_input("💬 Type your message"):
            st.session_state.chat_history.append(("🧑 You", prompt))

            messages = [{"role": "system", "content": (
                "You are WRITER PRO, an intelligent assistant developed by Mrs Abid. "
                "You are creative, accurate, and helpful in writing, coding, idea generation, and more.")}]
            for sender, content in st.session_state.chat_history:
                messages.append({"role": "user" if sender == "🧑 You" else "assistant", "content": content})

            with st.spinner("WRITER is thinking..."):
                response_container = st.empty()
                full_response = ""

                if stream := await get_ai_response(client, messages, model):
                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_container.markdown(
                                f'<div class="message-bot"><b>🤖 WRITER:</b> {full_response}</div>',
                                unsafe_allow_html=True
                            )

                    st.session_state.chat_history.append(("🤖 WRITER", full_response))
                    st.session_state.last_tokens = len(prompt.split()) + len(full_response.split())
                    save_chat(st.session_state.chat_history, model, st.session_state.last_tokens)
                    st.rerun()

    elif option == "📖 Chat Library":
        st.markdown("## 📚 Saved Chats")
        if os.path.exists("chats"):
            if files := [f for f in os.listdir("chats") if f.endswith(".json")]:
                selected = st.selectbox("Select a chat", sorted(files, reverse=True))
                with open(f"chats/{selected}", "r") as f:
                    data = json.load(f)
                    st.caption(f"Model: {data['metadata']['model']} | Date: {data['metadata']['created_at']} | Tokens: {data['metadata']['token_count']}")
                    display_chat(data["messages"])
            else:
                st.info("No saved chats yet.")

    elif option == "⚙️ Settings":
        st.markdown("## ⚙️ Settings")
        st.checkbox("Enable message streaming", value=True)
        st.checkbox("Save chat history", value=True)
        st.slider("Max tokens in response", 100, 2000, 500)

    if st.session_state.last_tokens and option in ["💬 Current Chat", "🌟 New Chat"]:
        st.markdown(f'<div class="token-counter">🔢 Tokens used: {st.session_state.last_tokens}</div>',
                    unsafe_allow_html=True)

if __name__ == "__main__":
    asyncio.run(main())
