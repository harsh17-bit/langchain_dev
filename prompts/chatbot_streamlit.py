import os
from pathlib import Path
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

st.set_page_config(page_title="Chat-Bot", page_icon=" ", layout="wide")

MODEL_CHOICES = [
    "command-a-03-2025",
    "command-r7b-12-2024",
    "command-r-plus-08-2024",
    "command-r-08-2024",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, high-signal AI assistant. Answer directly and practically. "
    "Ask one clarifying question only when needed. For coding requests, provide simple working examples."
)

THEME_TOKENS = {
    "Light": {
        "bg": "#f7f7f8",
        "surface": "#ffffff",
        "panel": "#ffffff",
        "border": "#e5e7eb",
        "text": "#1f2937",
        "muted": "#6b7280",
        "accent": "#10a37f",
    },
    "Dark": {
        "bg": "#212121",
        "surface": "#2a2a2a",
        "panel": "#303030",
        "border": "#3a3a3a",
        "text": "#ececec",
        "muted": "#a7a7a7",
        "accent": "#10a37f",
    },
}

HISTORY_FILE = Path(__file__).with_name("chathistory,txt")


class ChatItem(TypedDict):
    role: str
    content: str


def init_session_state() -> None:
    st.session_state.setdefault("theme", "Light")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("temperature", 0.3)
    st.session_state.setdefault("current_model_name", os.getenv("COHERE_MODEL", MODEL_CHOICES[0]))
    st.session_state.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)


def get_api_key() -> str | None:
    key = os.getenv("COHERE_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("COHERE_API_KEY")
    except Exception:
        return None


def build_theme_css(theme_name: str) -> str:
    palette = THEME_TOKENS[theme_name]
    return f"""
    <style>
    :root {{
        --bg: {palette['bg']};
        --surface: {palette['surface']};
        --panel: {palette['panel']};
        --border: {palette['border']};
        --text: {palette['text']};
        --muted: {palette['muted']};
        --accent: {palette['accent']};
    }}

    .stApp {{
        background: var(--bg);
        color: var(--text);
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    section[data-testid="stSidebar"] {{
        background: var(--surface);
        border-right: 1px solid var(--border);
    }}

    div[data-testid="stChatMessage"] {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
    }}

    .stButton > button,
    .stDownloadButton > button {{
        background: var(--panel);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: var(--accent);
    }}

    .stSelectbox div[data-baseweb="select"] > div,
    .stTextArea textarea,
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] input {{
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }}

    .empty-note {{
        color: var(--muted);
        margin: 1rem 0;
    }}
    </style>
    """


def build_model(model_name: str, api_key: str) -> ChatCohere:
    return ChatCohere(
        model=model_name,
        temperature=st.session_state.temperature,
        cohere_api_key=api_key,
    )


def to_langchain_messages(history: list[ChatItem]) -> list:
    mapped = [SystemMessage(content=st.session_state.system_prompt)]
    for item in history:
        if item["role"] == "user":
            mapped.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            mapped.append(AIMessage(content=item["content"]))
    return mapped


def save_history(history: list[ChatItem]) -> None:
    lines = []
    for item in history:
        if item["role"] == "user":
            lines.append(f'HumanMessage(content="{item["content"]}")')
        elif item["role"] == "assistant":
            lines.append(f'AIMessage(content="{item["content"]}")')
    HISTORY_FILE.write_text("\n".join(lines), encoding="utf-8")


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Cohere Chat")
        st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            key="theme_picker",
        )

        default_model = os.getenv("COHERE_MODEL", MODEL_CHOICES[0])
        st.selectbox(
            "Model",
            MODEL_CHOICES,
            index=MODEL_CHOICES.index(default_model) if default_model in MODEL_CHOICES else 0,
            key="model_picker",
        )
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            key="temperature_picker",
        )

        with st.expander("Assistant behavior", expanded=False):
            st.text_area(
                "Instructions",
                value=st.session_state.system_prompt,
                height=140,
                key="system_prompt_picker",
            )

        st.session_state.theme = st.session_state.theme_picker
        st.session_state.current_model_name = st.session_state.model_picker
        st.session_state.temperature = st.session_state.temperature_picker
        st.session_state.system_prompt = st.session_state.system_prompt_picker

        st.markdown("---")
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("Clear saved history", use_container_width=True):
            st.session_state.messages = []
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()
            st.rerun()


def render_chat_history() -> None:
    if not st.session_state.messages:
        st.markdown('<div class="empty-note">Start a conversation.</div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])


def process_user_prompt(api_key: str, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            model = build_model(st.session_state.current_model_name, api_key)
            request_messages = to_langchain_messages(st.session_state.messages)
            try:
                result = model.invoke(request_messages)
            except Exception as err:
                st.error(f"Request failed: {err}")
                st.stop()

            answer = result.content if isinstance(result.content, str) else str(result.content)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_history(st.session_state.messages)


def render_download_button() -> None:
    history_text = "\n".join([f"[{m['role'].upper()}] {m['content']}" for m in st.session_state.messages])
    with st.sidebar:
        st.download_button(
            "Download chat",
            data=history_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )


def main() -> None:
    init_session_state()

    api_key = get_api_key()
    if not api_key:
        st.error("COHERE_API_KEY not found. Add it to .env or Streamlit secrets.")
        st.stop()

    render_sidebar()
    st.markdown(build_theme_css(st.session_state.theme), unsafe_allow_html=True)
    render_chat_history()

    prompt = st.chat_input("Message Cohere Chat")
    if prompt:
        process_user_prompt(api_key, prompt)

    render_download_button()


if __name__ == "__main__":
    main()
