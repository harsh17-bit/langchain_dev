import os
import warnings
import logging
from pathlib import Path

# Silence noisy Transformers advisory warnings before model imports.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=r"Accessing `__path__` from")
logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

load_dotenv()

st.set_page_config(page_title="Cohere Research Tool", page_icon="🔎")
st.header("Reasearch Tool")

api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    st.error("COHERE_API_KEY not found. Add it in your .env file and restart Streamlit.")
    st.stop()

# Use explicit live model IDs instead of deprecated aliases.
MODEL_CANDIDATES = [
    "command-a-03-2025",
    "command-r7b-12-2024",
    "command-r-plus-08-2024",
]

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)

template_path = Path(__file__).with_name("template.json")
template = load_prompt(str(template_path))


def try_model_with_prompt(inputs: dict):
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            model = ChatCohere(model=model_name, cohere_api_key=api_key)
            chain = template | model
            result = chain.invoke(inputs)
            return result, model_name, None
        except Exception as err:
            last_error = err

    return None, None, last_error

if st.button("Summarize"):
    with st.spinner("Summarizing..."):
        result, model_name, err = try_model_with_prompt(
            {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input,
            }
        )

    if err is not None:
        st.error(f"Request failed: {err}")
        st.info("Check your API key access and try a model listed in Cohere docs.")
    else:
        st.caption(f"Model used: {model_name}")
        st.write(result.content)