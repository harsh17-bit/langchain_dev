import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

st.set_page_config(
    page_title="Cohere Chat Studio",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f7f5ef;
        --card: #fffdf8;
        --ink: #10231d;
        --accent: #0f766e;
        --accent-2: #d97706;
        --border: #d6d3c5;
        --muted: #5c6b65;
    }

    .stApp {
        background:
          radial-gradient(circle at 10% 10%, #e9f4ef 0%, transparent 45%),
          radial-gradient(circle at 90% 5%, #fef1d4 0%, transparent 38%),
          var(--bg);
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }

    .top-banner {
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        background: linear-gradient(145deg, #fffef8 0%, #f8f6ee 100%);
        box-shadow: 0 10px 30px rgba(16, 35, 29, 0.06);
    }

    .top-banner h2 {
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.2px;
    }

    .top-banner p {
        margin-top: 0.3rem;
        color: var(--muted);
    }

    .small-pill {
        border: 1px solid var(--border);
        background: var(--card);
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        display: inline-block;
        margin-right: 0.5rem;
        color: var(--muted);
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-banner">
      <h2>Cohere Chat Studio</h2>
      <p>Conversation UI for your LangChain + Cohere chatbot with model fallback and history export.</p>
      <span class="small-pill">LangChain</span>
      <span class="small-pill">Cohere</span>
      <span class="small-pill">Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("COHERE_API_KEY") or st.secrets.get("COHERE_API_KEY")
if not api_key:
    st.error("COHERE_API_KEY not found. Add it to .env or Streamlit secrets.")
    st.stop()

HISTORY_FILE = Path(__file__).with_name("chathistory,txt")

MODEL_CHOICES = [
    "command-a-03-2025",
    "command-r7b-12-2024",
    "command-r-plus-08-2024",
    "command-r-08-2024",
]

with st.sidebar:
    st.subheader("Model Settings")
    default_primary = os.getenv("COHERE_MODEL", "command-a-03-2025")
    default_fallback = os.getenv("COHERE_FALLBACK_MODEL", "command-r7b-12-2024")

    primary_model = st.selectbox(
        "Primary model",
        MODEL_CHOICES,
        index=MODEL_CHOICES.index(default_primary) if default_primary in MODEL_CHOICES else 0,
    )
    fallback_model = st.selectbox(
        "Fallback model",
        MODEL_CHOICES,
        index=MODEL_CHOICES.index(default_fallback) if default_fallback in MODEL_CHOICES else 1,
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
        st.session_state.current_model_name = primary_model
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

if "current_model_name" not in st.session_state:
    st.session_state.current_model_name = primary_model


def build_model(model_name: str) -> ChatCohere:
    return ChatCohere(
        model=model_name,
        temperature=temperature,
        cohere_api_key=api_key,
    )


def to_langchain_messages(history: list[dict]):
    mapped = []
    for item in history:
        role = item["role"]
        text = item["content"]
        if role == "system":
            mapped.append(SystemMessage(content=text))
        elif role == "user":
            mapped.append(HumanMessage(content=text))
        elif role == "assistant":
            mapped.append(AIMessage(content=text))
    return mapped


def save_history(history: list[dict]) -> None:
    lines = []
    for item in history:
        if item["role"] == "system":
            continue
        if item["role"] == "user":
            lines.append(f'HumanMessage(content="{item["content"]}")')
        elif item["role"] == "assistant":
            lines.append(f'AIMessage(content="{item["content"]}")')

    HISTORY_FILE.write_text("\n".join(lines), encoding="utf-8")


for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            model = build_model(st.session_state.current_model_name)
            request_messages = to_langchain_messages(st.session_state.messages)

            try:
                result = model.invoke(request_messages)
            except Exception as err:
                error_text = str(err)
                should_fallback = (
                    st.session_state.current_model_name != fallback_model
                    and (
                        "status_code: 404" in error_text
                        or "was removed" in error_text
                        or "not found" in error_text.lower()
                    )
                )

                if should_fallback:
                    st.warning(
                        f"Model '{st.session_state.current_model_name}' unavailable. Switching to '{fallback_model}'."
                    )
                    st.session_state.current_model_name = fallback_model
                    model = build_model(st.session_state.current_model_name)
                    result = model.invoke(request_messages)
                else:
                    st.error(f"Request failed: {err}")
                    st.stop()

            answer = result.content if isinstance(result.content, str) else str(result.content)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_history(st.session_state.messages)

history_text = "\n".join(
    [
        f"[{m['role'].upper()}] {m['content']}"
        for m in st.session_state.messages
        if m["role"] != "system"
    ]
)

c1, c2 = st.columns([2, 1])
with c1:
    st.caption(
        f"Current model: {st.session_state.current_model_name} | Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
with c2:
    st.download_button(
        "Download chat",
        data=history_text,
        file_name="chat_history.txt",
        mime="text/plain",
        use_container_width=True,
    )
