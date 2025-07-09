# -------------------------------------------------------------
# Streamlit Next‑Word Predictor (simple, no auto‑append)
# -------------------------------------------------------------
# * File watcher disabled to avoid torch.classes crash.
# * Suggestions shown as colorful chips — user can type them manually.
# * Example prompts dropdown.
# -------------------------------------------------------------

import os, textwrap

# Disable Streamlit file‑watcher which causes torch.classes crash
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ----------------------- CONFIG --------------------------------
MODEL_DIR = "SillySom/next-word-predictor"   # path to your fine‑tuned model
TOP_K     = 10                       # number of suggestions to show
EXAMPLE_PROMPTS = [
    "Following the departure of Ben Foster from Manchester United",
    "The Office of Military Cooperation",
    "At the conclusion of the regular season",
    "An oxaziridine is an",
    "The campaign in the upper Danube valley began in",
    "The Civil War saw",
]
# ----------------------------------------------------------------

st.set_page_config(page_title="Next‑Word Predictor", page_icon="✨", layout="centered")

# Inject a little CSS for visible, styled chips
st.markdown(
    """
    <style>
    .chip {
        display:inline-block;
        padding:0.4rem 0.8rem;
        margin:0.3rem 0.4rem;
        background:#f1f3f5;
        color:#212529;
        border-radius:16px;
        font-size:0.95rem;
        font-weight:500;
        border:1px solid #ced4da;
        box-shadow:0 1px 2px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading fine‑tuned model…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model, device

TOKENIZER, MODEL, DEVICE = load_model()

# ---------------------- Helper ----------------------------------

def get_top_k_tokens(prompt: str, k: int = TOP_K):
    enc = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = MODEL(**enc).logits
    probs = softmax(logits[0, -1], dim=-1)
    top = probs.topk(k)
    return [TOKENIZER.decode(tid).lstrip(" ") for tid in top.indices]

# ---------------------- UI --------------------------------------

st.title("✨ Next‑Word Predictor")

# Example prompt selector
example = st.selectbox("Choose an example prompt (optional):", ["(none)"] + EXAMPLE_PROMPTS)

prompt = st.text_area("Prompt", value="" if example == "(none)" else example, height=120)

if st.button("Suggest next words"):
    if not prompt.strip():
        st.error("Please enter a prompt first.")
    else:
        suggestions = get_top_k_tokens(prompt.strip())
        st.markdown("**Top suggestions:**")
        chips_html = "".join(f'<span class="chip">{tok}</span>' for tok in suggestions)
        st.markdown(f"<div style='line-height: 2.2;'>{chips_html}</div>", unsafe_allow_html=True)

st.markdown(
    "<small>Type one of the suggested words manually after your prompt and click again to get the next set of suggestions.</small>",
    unsafe_allow_html=True,
)
