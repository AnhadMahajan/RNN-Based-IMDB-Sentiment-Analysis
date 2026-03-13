import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.keras')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,700;1,400;1,700&family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #0c0c0c;
    color: #c8c4bc;
}

.block-container {
    padding: 3rem 3rem 4rem 3rem;
    max-width: 100%;
}

section[data-testid="stSidebar"] {
    background-color: #0e0e0e;
    border-right: 1px solid #1e1e1e;
}

section[data-testid="stSidebar"] > div {
    padding: 2rem 1.4rem;
}

.sidebar-wordmark {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1a1a1a;
}

.sidebar-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 1.2rem;
    margin-top: 2rem;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.7rem 0;
    border-bottom: 1px solid #181818;
}

.stat-row:last-child { border-bottom: none; }

.stat-key {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 300;
    color: #7a7a7a;
}

.stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #a8a49c;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid #1a1a1a;
    margin: 1.8rem 0;
}

.note-block {
    background-color: #111111;
    border-left: 2px solid #2a2420;
    padding: 0.9rem 1rem;
    margin-top: 1.5rem;
}

.note-text {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 300;
    color: #7a7268;
    line-height: 1.7;
}

.page-header {
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 2.2rem;
    margin-bottom: 2.8rem;
}

.header-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.8rem;
}

.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 700;
    color: #ede8de;
    line-height: 1.1;
    margin: 0 0 0.8rem 0;
    letter-spacing: -0.02em;
}

.header-title em {
    font-style: italic;
    color: #c8a96e;
}

.header-desc {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 300;
    color: #7a7a7a;
    max-width: 560px;
    line-height: 1.75;
}

div[data-testid="stTextArea"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.58rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #666 !important;
}

div[data-testid="stTextArea"] textarea {
    background-color: #111111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 2px !important;
    color: #bab6ae !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 300 !important;
    line-height: 1.8 !important;
    caret-color: #c8a96e !important;
    padding: 1.2rem 1.4rem !important;
    transition: border-color 0.25s ease !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #2e2e2e !important;
    box-shadow: none !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: #252525 !important;
    font-style: italic;
}

div.stButton > button {
    background-color: transparent !important;
    border: 1px solid #c8a96e !important;
    border-radius: 2px !important;
    color: #c8a96e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2.4rem !important;
    transition: background-color 0.2s ease, color 0.2s ease !important;
}

div.stButton > button:hover {
    background-color: #c8a96e !important;
    color: #0c0c0c !important;
}

.meta-row {
    display: flex;
    gap: 2.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #181818;
}

.meta-item {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.meta-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #555;
}

.meta-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #888;
}

.result-panel {
    background-color: #0f0f0f;
    border: 1px solid #1c1c1c;
    padding: 2.4rem 2.6rem;
    min-height: 380px;
    display: flex;
    flex-direction: column;
}

.result-panel-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #1a1a1a;
}

.verdict-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 0.5rem 0 2rem 0;
}

.verdict-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.verdict-tag-positive { color: #6a6040; }
.verdict-tag-negative { color: #4a4040; }

.verdict-word {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 0.9;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}

.verdict-positive { color: #ddd4b4; }
.verdict-negative { color: #7a6a5a; }
.verdict-italic { font-style: italic; font-weight: 400; }

.score-section {
    border-top: 1px solid #1a1a1a;
    padding-top: 1.6rem;
}

.score-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.score-row:last-child { margin-bottom: 0; }

.score-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    min-width: 100px;
}

.score-track {
    flex: 1;
    height: 2px;
    background-color: #1a1a1a;
    border-radius: 1px;
    overflow: hidden;
}

.score-fill-positive {
    height: 100%;
    background-color: #c8a96e;
    border-radius: 1px;
}

.score-fill-negative {
    height: 100%;
    background-color: #4a4040;
    border-radius: 1px;
}

.score-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    min-width: 50px;
    text-align: right;
}

.breakdown-panel {
    background-color: #0f0f0f;
    border: 1px solid #1c1c1c;
    border-top: none;
    padding: 1.4rem 2.6rem;
}

.breakdown-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
}

.breakdown-cell {
    padding: 0.6rem 1.4rem 0.6rem 0;
    border-right: 1px solid #1a1a1a;
    margin-right: 1.4rem;
}

.breakdown-cell:last-child {
    border-right: none;
    margin-right: 0;
}

.breakdown-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.52rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.4rem;
}

.breakdown-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #999;
}

.idle-panel {
    background-color: #0f0f0f;
    border: 1px solid #1c1c1c;
    padding: 3.5rem 2.6rem;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    min-height: 380px;
}

.idle-number {
    font-family: 'Playfair Display', serif;
    font-size: 7rem;
    font-weight: 700;
    color: #2a2a2a;
    line-height: 1;
    margin-bottom: 1.5rem;
    letter-spacing: -0.04em;
    font-style: italic;
}

.idle-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.6rem;
}

.idle-desc {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 300;
    color: #555;
    line-height: 1.8;
    max-width: 240px;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-wordmark">Sentiment / IMDB &mdash; v1.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <div class="stat-row"><span class="stat-key">Architecture</span><span class="stat-val">Simple RNN</span></div>
        <div class="stat-row"><span class="stat-key">Framework</span><span class="stat-val">Keras / TF</span></div>
        <div class="stat-row"><span class="stat-key">Activation</span><span class="stat-val">ReLU</span></div>
        <div class="stat-row"><span class="stat-key">Input length</span><span class="stat-val">500 tokens</span></div>
        <div class="stat-row"><span class="stat-key">Output</span><span class="stat-val">Sigmoid [0,1]</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <div class="stat-row"><span class="stat-key">Source</span><span class="stat-val">IMDB</span></div>
        <div class="stat-row"><span class="stat-key">Train samples</span><span class="stat-val">25,000</span></div>
        <div class="stat-row"><span class="stat-key">Test samples</span><span class="stat-val">25,000</span></div>
        <div class="stat-row"><span class="stat-key">Vocabulary</span><span class="stat-val">88,587 words</span></div>
        <div class="stat-row"><span class="stat-key">Classes</span><span class="stat-val">2 (binary)</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Classification Rule</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <div class="stat-row"><span class="stat-key">Positive</span><span class="stat-val">score &gt; 0.5</span></div>
        <div class="stat-row"><span class="stat-key">Negative</span><span class="stat-val">score &le; 0.5</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="note-block">
        <div class="note-text">
            Unknown words map to index&nbsp;2. Input is truncated or zero-padded to 500 tokens before inference.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="header-eyebrow">Recurrent Neural Network &mdash; IMDB Corpus &mdash; Binary Classification</div>
    <div class="header-title">Movie Review<br><em>Sentiment</em></div>
    <div class="header-desc">
        Enter any movie review below. The model tokenises your input against the IMDB vocabulary, pads it to 500 tokens,
        and returns a continuous probability score with a binary sentiment classification.
    </div>
</div>
""", unsafe_allow_html=True)

col_input, col_result = st.columns([1.05, 0.95], gap="large")

with col_input:
    user_input = st.text_area(
        "Review Text",
        height=300,
        placeholder="The film was a masterclass in tension — every scene carefully constructed, the performances restrained yet devastating...",
        label_visibility="visible"
    )

    word_count = len(user_input.strip().split()) if user_input.strip() else 0
    token_count = min(word_count, 500)

    st.markdown(f"""
    <div class="meta-row">
        <div class="meta-item">
            <span class="meta-key">Words</span>
            <span class="meta-value">{word_count}</span>
        </div>
        <div class="meta-item">
            <span class="meta-key">Tokens (capped at 500)</span>
            <span class="meta-value">{token_count} / 500</span>
        </div>
        <div class="meta-item">
            <span class="meta-key">Characters</span>
            <span class="meta-value">{len(user_input)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True)
    classify = st.button("Run Analysis")

with col_result:
    if classify:
        if user_input.strip():
            with st.spinner(""):
                preprocessed_input = preprocess_text(user_input)
                prediction = model.predict(preprocessed_input)
                score = float(prediction[0][0])
                sentiment = "Positive" if score > 0.5 else "Negative"
                confidence = score if score > 0.5 else 1 - score
                fill_class = "score-fill-positive" if sentiment == "Positive" else "score-fill-negative"
                verdict_class = "verdict-positive" if sentiment == "Positive" else "verdict-negative"
                tag_class = "verdict-tag-positive" if sentiment == "Positive" else "verdict-tag-negative"
                margin = abs(score - 0.5)

            st.markdown(f"""
            <div class="result-panel">
                <div class="result-panel-header">Classification Result</div>
                <div class="verdict-container">
                    <div class="verdict-tag {tag_class}">Sentiment Detected</div>
                    <div class="verdict-word {verdict_class}">{sentiment[:3]}<span class="verdict-italic">{sentiment[3:]}</span></div>
                </div>
                <div class="score-section">
                    <div class="score-row">
                        <span class="score-label">Confidence</span>
                        <div class="score-track"><div class="{fill_class}" style="width:{confidence*100:.1f}%"></div></div>
                        <span class="score-value">{confidence*100:.1f}%</span>
                    </div>
                    <div class="score-row">
                        <span class="score-label">Positive prob.</span>
                        <div class="score-track"><div class="{fill_class}" style="width:{score*100:.1f}%"></div></div>
                        <span class="score-value">{score:.4f}</span>
                    </div>
                    <div class="score-row">
                        <span class="score-label">Negative prob.</span>
                        <div class="score-track"><div class="score-fill-negative" style="width:{(1-score)*100:.1f}%"></div></div>
                        <span class="score-value">{1-score:.4f}</span>
                    </div>
                </div>
            </div>
            <div class="breakdown-panel">
                <div class="breakdown-grid">
                    <div class="breakdown-cell">
                        <div class="breakdown-key">Raw Score</div>
                        <div class="breakdown-val">{score:.6f}</div>
                    </div>
                    <div class="breakdown-cell">
                        <div class="breakdown-key">Decision</div>
                        <div class="breakdown-val">{sentiment}</div>
                    </div>
                    <div class="breakdown-cell">
                        <div class="breakdown-key">Margin from 0.5</div>
                        <div class="breakdown-val">{margin:.4f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="idle-panel">
                <div class="idle-number">!</div>
                <div class="idle-label">Input Required</div>
                <div class="idle-desc">Enter a movie review on the left before running analysis.</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="idle-panel">
            <div class="idle-number">—</div>
            <div class="idle-label">Awaiting Input</div>
            <div class="idle-desc">Write or paste a movie review, then press Run Analysis to classify its sentiment.</div>
        </div>
        """, unsafe_allow_html=True)