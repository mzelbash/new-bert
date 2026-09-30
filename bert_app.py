"""
BERT: From Pretraining to Fine-tuning
SEAS 8525 - Computer Vision and Generative AI
Dr. Elbasheer

Run:  streamlit run bert_app.py
Theme: put config.toml in a folder named .streamlit next to this file.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(
    page_title="BERT Walkthrough",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette (one color code for the whole app) ────────────────────────────────
NAVY   = "#002147"
GOLD   = "#FFC400"
BLUE   = "#1D4ED8"   # inputs and tokens (sentence A)
PURPLE = "#7C3AED"   # sentence B
RED    = "#E11D48"   # [MASK] and hidden states
GREEN  = "#0E9F6E"   # outputs and predictions
ORANGE = "#D97706"   # losses and training signal
MUTED  = "#475569"

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root{
  --navy:#002147; --gold:#FFC400; --bg:#F3F6FB; --border:#D9E1EC;
  --text:#0F172A; --muted:#475569;
  --blue:#1D4ED8; --purple:#7C3AED; --red:#E11D48; --green:#0E9F6E; --orange:#D97706;
}

/* ── App shell ── */
.stApp{ background:var(--bg); color:var(--text); }
[data-testid="stHeader"]{ background:transparent; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp div, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3{
  font-family:'Lexend', system-ui, sans-serif;
}
.stApp code, .stApp pre, .stApp pre *{ font-family:'JetBrains Mono', monospace !important; }
[data-testid="stMainBlockContainer"], .block-container{
  max-width:1180px; padding-top:2.2rem; padding-bottom:3rem;
}
[data-testid="stSidebar"]{ background:#FFFFFF; border-right:1px solid var(--border); }

/* ── Widgets ── */
.stApp label p{ font-size:1rem !important; font-weight:600; color:var(--navy); }
.stTextInput input, .stTextArea textarea{
  font-size:1.05rem !important; border-radius:10px !important;
}
.stButton > button{ border-radius:10px; font-weight:700; font-size:1rem; padding:.55rem 1.3rem; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--navy) !important; border:none !important; color:#fff !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
  background:#0B3470 !important; color:var(--gold) !important;
}
[data-testid="stExpander"] details{
  background:#fff; border:1px solid var(--border) !important; border-radius:14px !important;
}
[data-testid="stExpander"] summary p{ font-weight:600; color:var(--navy); font-size:1rem; }
[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:16px; }

/* Sidebar radio as a clean menu */
[data-testid="stSidebar"] [role="radiogroup"]{ gap:2px; }
[data-testid="stSidebar"] [role="radiogroup"] label{
  padding:8px 10px; border-radius:10px; width:100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{ background:#F3F6FB; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){ background:#EAF0FA; }
[data-testid="stSidebar"] [role="radiogroup"] label p{
  font-size:.95rem !important; font-weight:500; color:#1E293B;
}

/* ── Section header ── */
.sec-hdr{ margin:0 0 22px; }
.sec-pill{ display:inline-block; background:var(--gold); color:var(--navy); font-weight:800;
  font-size:.78rem; letter-spacing:.12em; text-transform:uppercase;
  padding:5px 13px; border-radius:999px; }
.sec-title{ font-size:2.55rem; font-weight:800; color:var(--navy); letter-spacing:-.02em;
  line-height:1.12; margin:14px 0 8px; }
.sec-sub{ font-size:1.18rem; color:var(--muted); line-height:1.5; }
.sec-rule{ height:5px; width:80px; background:var(--gold); border-radius:4px; margin-top:16px; }

/* ── Main idea ── */
.key{ background:var(--navy); border-left:9px solid var(--gold); border-radius:18px;
  padding:24px 30px; margin:4px 0 28px; box-shadow:0 10px 26px rgba(0,33,71,.16); }
.key-label{ color:var(--gold); font-size:.8rem; font-weight:800; letter-spacing:.16em;
  text-transform:uppercase; }
.key-text{ color:#fff; font-size:1.45rem; font-weight:600; line-height:1.45; margin-top:8px; }
.key-text b{ color:var(--gold); font-weight:700; }

/* ── Body text ── */
.lead{ font-size:1.13rem; line-height:1.75; color:#1E293B; margin:0 0 22px; }
.lead b{ color:var(--navy); }
.hint{ font-size:1.02rem; color:var(--muted); line-height:1.6; margin:0 0 14px; }
.sub{ display:flex; align-items:center; gap:12px; font-size:1.55rem; font-weight:800;
  color:var(--navy); margin:38px 0 16px; letter-spacing:-.01em; }
.sub::before{ content:""; width:10px; height:28px; background:var(--gold); border-radius:3px; }

/* ── Grids and cards ── */
.grid{ display:grid; gap:18px; margin:4px 0 20px; }
.g2{ grid-template-columns:repeat(2,1fr); }
.g3{ grid-template-columns:repeat(3,1fr); }
@media (max-width:900px){ .g2,.g3{ grid-template-columns:1fr; } }
.card{ background:#fff; border:1px solid var(--border); border-top:6px solid var(--c);
  border-radius:16px; padding:18px 22px; }
.card-title{ font-size:1.18rem; font-weight:700; color:var(--c); margin-bottom:8px; }
.card-body{ font-size:1.02rem; line-height:1.65; color:#334155; }
.card-body b{ color:var(--text); }
.card code{ font-size:.9rem; background:#F1F5F9; padding:2px 6px; border-radius:6px; color:var(--navy); }

/* ── Tags ── */
.tag{ display:inline-block; padding:3px 11px; border-radius:999px; font-size:.84rem;
  font-weight:700; color:var(--c); background:color-mix(in srgb, var(--c) 13%, white);
  border:1px solid color-mix(in srgb, var(--c) 35%, white); margin:2px 4px 2px 0;
  white-space:nowrap; }

/* ── Class tip ── */
.tip{ background:#FFFBEA; border:1px solid #FDE68A; border-left:7px solid var(--gold);
  border-radius:14px; padding:16px 22px; margin:24px 0; font-size:1.04rem; line-height:1.7;
  color:#1E293B; }
.tip-label{ display:inline-block; background:var(--navy); color:var(--gold); font-size:.72rem;
  font-weight:800; letter-spacing:.12em; text-transform:uppercase; padding:3px 10px;
  border-radius:999px; margin-right:8px; }

/* ── Tables ── */
.tbl-wrap{ background:#fff; border:1px solid var(--border); border-radius:16px;
  overflow:hidden; overflow-x:auto; margin:4px 0 20px; box-shadow:0 2px 8px rgba(0,33,71,.05); }
.tbl{ width:100%; border-collapse:collapse; font-size:1.02rem; }
.tbl th{ background:var(--navy); color:#fff; font-weight:700; text-align:left;
  padding:14px 18px; font-size:.95rem; letter-spacing:.02em; }
.tbl td{ padding:14px 18px; border-top:1px solid #E8EDF4; vertical-align:top;
  line-height:1.55; color:#1E293B; }
.tbl tbody tr:nth-child(even) td{ background:#F7F9FC; }
.tbl td:first-child{ font-weight:700; color:var(--navy); }
.tbl tr.hl td{ background:#FFF6D1 !important; }
.tbl td.ncol{ font-family:'JetBrains Mono', monospace; font-weight:700; text-align:center; }
.tbl th.ncol{ text-align:center; }
.tbl .mono{ font-family:'JetBrains Mono', monospace; font-size:.92rem; color:var(--navy);
  white-space:nowrap; }

/* ── Token pills ── */
.tok-row{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:12px 0; }
.tok{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1rem;
  padding:6px 12px; border-radius:10px; border:2px solid; line-height:1.3; }
.tok.cls { background:var(--gold); color:var(--navy); border-color:var(--gold); }
.tok.sep { background:#E2E8F0; color:var(--navy); border-color:#94A3B8; }
.tok.mask{ background:#FFE4EA; color:var(--red); border-color:var(--red); }
.tok.a   { background:#DBEAFE; color:var(--blue); border-color:#93C5FD; }
.tok.b   { background:#EDE9FE; color:var(--purple); border-color:#C4B5FD; }
.tok.out { background:#D1FAE5; color:var(--green); border-color:var(--green); }
.tok.wp  { border-style:dashed; }
.tok-legend{ display:flex; flex-wrap:wrap; gap:16px; font-size:.92rem; color:var(--muted);
  margin:6px 0 16px; }
.tok-legend span.sw{ display:inline-block; width:14px; height:14px; border-radius:4px;
  margin-right:6px; vertical-align:-2px; border:2px solid; }

/* ── Numbered steps ── */
.steps{ background:#fff; border:1px solid var(--border); border-radius:16px; padding:6px 22px; }
.step{ display:flex; gap:16px; align-items:flex-start; padding:14px 0; }
.step + .step{ border-top:1px dashed var(--border); }
.num{ flex:0 0 34px; height:34px; border-radius:50%; background:var(--gold); color:var(--navy);
  font-weight:800; display:flex; align-items:center; justify-content:center; font-size:1rem; }
.step-text{ font-size:1.05rem; line-height:1.6; color:#1E293B; padding-top:4px; }
.step-text b{ color:var(--navy); }

/* ── Flow (Section 1) ── */
.flow{ display:flex; align-items:stretch; gap:10px; margin:6px 0 26px; flex-wrap:wrap; }
.flow-box{ flex:1 1 220px; background:#fff; border:1px solid var(--border);
  border-top:6px solid var(--c); border-radius:16px; padding:18px 20px; }
.flow-kicker{ font-size:.75rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
  color:var(--c); }
.flow-title{ font-size:1.25rem; font-weight:800; color:var(--navy); margin:6px 0 10px; }
.flow-box ul{ margin:0; padding-left:18px; font-size:1rem; line-height:1.75; color:#334155; }
.flow-op{ display:flex; align-items:center; font-size:2.2rem; font-weight:800; color:var(--navy);
  padding:0 2px; }

/* ── Equation boxes (Section 2) ── */
.eq{ display:flex; align-items:stretch; gap:10px; flex-wrap:wrap; margin:6px 0 10px; }
.eq-box{ flex:1 1 170px; border-radius:14px; padding:16px 16px; text-align:center;
  background:color-mix(in srgb, var(--c) 10%, white); border:2px solid var(--c); }
.eq-name{ font-weight:800; font-size:1.1rem; color:var(--c); }
.eq-desc{ font-size:.95rem; color:#334155; margin-top:6px; line-height:1.45; }
.eq-op{ display:flex; align-items:center; font-size:2rem; font-weight:800; color:var(--navy); }
.eq-note{ text-align:center; font-size:1rem; color:var(--muted); margin:6px 0 4px; }

/* ── Worked embedding example (Section 2) ── */
.eg-wrap{ background:#fff; border:1px solid var(--border); border-radius:16px; padding:18px 18px;
  overflow-x:auto; margin:4px 0 8px; }
.eg{ display:grid; gap:8px 6px; align-items:center; min-width:820px; }
.eg-label{ font-weight:800; font-size:.98rem; text-align:right; padding-right:10px; }
.eg-col{ display:flex; justify-content:center; }
.eg-col .tok{ font-size:.85rem; padding:5px 8px; }
.eg-op{ font-size:1.3rem; font-weight:800; color:var(--navy); text-align:right; padding-right:14px;
  line-height:.8; }
.chip{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:.85rem; text-align:center;
  padding:7px 2px; border-radius:9px; color:var(--c); border:2px solid var(--c);
  background:color-mix(in srgb, var(--c) 10%, white); }

/* ── Split bar (80/10/10) ── */
.split{ display:flex; height:54px; border-radius:12px; overflow:hidden; margin:10px 0 6px;
  border:1px solid var(--border); }
.split div{ display:flex; align-items:center; justify-content:center; color:#fff;
  font-weight:700; font-size:.95rem; text-align:center; padding:0 6px; line-height:1.2; }

/* ── Cross-attention diagram ── */
.xattn{ display:grid; grid-template-columns:1fr auto 1.2fr auto 1fr; gap:10px;
  align-items:center; margin:8px 0 20px; }
@media (max-width:900px){ .xattn{ grid-template-columns:1fr; } .xattn .arrow{ transform:rotate(90deg); } }
.x-box{ background:#fff; border:2px solid var(--c); border-radius:14px; padding:16px;
  text-align:center; }
.x-title{ font-weight:800; color:var(--c); font-size:1.08rem; }
.x-desc{ font-size:.95rem; color:#334155; margin-top:6px; line-height:1.5; }
.x-core{ background:var(--navy); color:#fff; border-radius:14px; padding:18px; text-align:center; }
.x-core .f{ font-family:'JetBrains Mono', monospace; font-size:1.02rem; color:var(--gold);
  margin-top:6px; }
.arrow{ font-size:2rem; font-weight:800; color:var(--navy); text-align:center; }

/* ── FAQ ── */
.qa{ background:#fff; border:1px solid var(--border); border-radius:14px; margin:0 0 12px;
  overflow:hidden; }
.qa-q{ padding:13px 18px; font-weight:700; font-size:1.03rem; color:var(--navy);
  background:#F7F9FC; }
.qa-q span{ background:var(--gold); color:var(--navy); border-radius:6px; padding:1px 8px;
  font-size:.8rem; margin-right:8px; }
.qa-a{ padding:13px 18px; font-size:1.02rem; line-height:1.7; color:#1E293B; }

/* ── Sidebar blocks ── */
.brand{ background:var(--navy); border-radius:16px; padding:18px 18px 16px; margin:4px 0 18px; }
.brand-pill{ display:inline-block; background:var(--gold); color:var(--navy); font-weight:800;
  font-size:.72rem; letter-spacing:.12em; padding:3px 10px; border-radius:999px; }
.brand-title{ color:#fff; font-size:1.35rem; font-weight:800; margin:10px 0 4px; }
.brand-course{ color:#CBD5E1; font-size:.9rem; }
.brand-inst{ color:var(--gold); font-size:.88rem; font-weight:600; margin-top:6px; }
.side-label{ font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
  color:#64748B; margin:18px 0 8px; }
.prog-text{ font-size:.9rem; font-weight:600; color:var(--navy); margin-bottom:6px; }
.prog{ height:8px; background:#E2E8F0; border-radius:99px; overflow:hidden; }
.prog div{ height:100%; background:var(--gold); border-radius:99px; }
.legend-item{ display:flex; align-items:center; gap:10px; font-size:.9rem; color:#1E293B;
  margin:6px 0; }
.legend-item i{ width:14px; height:14px; border-radius:4px; background:var(--c);
  display:inline-block; }
.setup-cmd{ display:block; font-family:'JetBrains Mono', monospace; font-size:.78rem;
  color:var(--navy); background:#F3F6FB; border:1px solid var(--border); border-radius:8px;
  padding:6px 10px; margin-top:6px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── HTML helpers ──────────────────────────────────────────────────────────────
def html(s):
    """Render raw HTML. Leading spaces and blank lines are removed so the
    markdown parser never turns indented HTML into a code block."""
    clean = "\n".join(line.strip() for line in s.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)

def section_header(num, total, title, subtitle):
    html(f"""
    <div class="sec-hdr">
      <span class="sec-pill">Section {num} of {total}</span>
      <div class="sec-title">{title}</div>
      <div class="sec-sub">{subtitle}</div>
      <div class="sec-rule"></div>
    </div>""")

def key_idea(text):
    html(f'<div class="key"><div class="key-label">Main idea</div>'
         f'<div class="key-text">{text}</div></div>')

def lead(text):
    html(f'<p class="lead">{text}</p>')

def hint(text):
    html(f'<p class="hint">{text}</p>')

def sub(text):
    html(f'<div class="sub">{text}</div>')

def tip(text):
    html(f'<div class="tip"><span class="tip-label">Class tip</span>{text}</div>')

def tag(text, color):
    return f'<span class="tag" style="--c:{color};">{text}</span>'

def card(title, body, color):
    return (f'<div class="card" style="--c:{color};">'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-body">{body}</div></div>')

def grid(cards, cols=2):
    html(f'<div class="grid g{cols}">{"".join(cards)}</div>')

def steps(items):
    rows = "".join(
        f'<div class="step"><div class="num">{i}</div><div class="step-text">{t}</div></div>'
        for i, t in enumerate(items, 1))
    html(f'<div class="steps">{rows}</div>')

def table(headers, rows, num_cols=(), highlight_rows=()):
    """headers: list of str. rows: list of lists (cells may contain HTML)."""
    th = "".join(f'<th class="{"ncol" if i in num_cols else ""}">{h}</th>'
                 for i, h in enumerate(headers))
    body = ""
    for r, row in enumerate(rows):
        tds = "".join(f'<td class="{"ncol" if i in num_cols else ""}">{c}</td>'
                      for i, c in enumerate(row))
        body += f'<tr class="{"hl" if r in highlight_rows else ""}">{tds}</tr>'
    html(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{th}</tr></thead>'
         f'<tbody>{body}</tbody></table></div>')

def tok(text, kind):
    return f'<span class="tok {kind}">{text}</span>'

def token_kind(token, segment):
    if token == "[CLS]":
        return "cls"
    if token == "[SEP]":
        return "sep"
    if token == "[MASK]":
        return "mask"
    base = "a" if segment == 0 else "b"
    return base + (" wp" if token.startswith("##") else "")

def token_row(tokens, segments):
    pills = "".join(tok(t, token_kind(t, s)) for t, s in zip(tokens, segments))
    html(f'<div class="tok-row">{pills}</div>')

def token_legend(show_b=True):
    items = [("#FFC400", "#FFC400", "[CLS] summary token"),
             ("#E2E8F0", "#94A3B8", "[SEP] boundary"),
             ("#DBEAFE", "#93C5FD", "Sentence A")]
    if show_b:
        items.append(("#EDE9FE", "#C4B5FD", "Sentence B"))
    items.append(("#FFFFFF", "#64748B", "##piece (dashed) = part of a word"))
    spans = "".join(
        f'<span><span class="sw" style="background:{bg};border-color:{bd};'
        f'{"border-style:dashed;" if "dashed" in lab else ""}"></span>{lab}</span>'
        for bg, bd, lab in items)
    html(f'<div class="tok-legend">{spans}</div>')

def faq(items):
    blocks = "".join(
        f'<div class="qa"><div class="qa-q"><span>Q{i}</span>{q}</div>'
        f'<div class="qa-a">{a}</div></div>' for i, (q, a) in enumerate(items, 1))
    html(blocks)


# ── Chart style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": "#D9E1EC",
    "axes.labelcolor": MUTED,
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
    "axes.titlesize": 13,
    "xtick.color": MUTED,
    "ytick.color": "#1E293B",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ── Model loading (imports are lazy so the app opens instantly) ───────────────
@st.cache_resource(show_spinner="Loading BERT tokenizer...")
def load_tokenizer():
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained("bert-base-uncased")

@st.cache_resource(show_spinner="Loading BERT model (about 30 s the first time)...")
def load_bert():
    from transformers import BertModel
    return BertModel.from_pretrained("bert-base-uncased")

@st.cache_resource(show_spinner="Loading fill-mask pipeline...")
def load_fill_mask():
    from transformers import pipeline
    return pipeline("fill-mask", model="bert-base-uncased")

@st.cache_resource(show_spinner="Loading sentiment pipeline...")
def load_sentiment():
    from transformers import pipeline
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")


# ── Navigation ────────────────────────────────────────────────────────────────
SECTIONS = [
    "The Bridge: Transformer to BERT",
    "BERT's Input: Three Embeddings",
    "Pretraining 1: Masked LM",
    "Pretraining 2: Next Sentence",
    "The [CLS] Token",
    "Fine-tuning for Tasks",
    "Encoder Output to a Decoder",
    "BERT Variants",
]
LABELS = [f"{i}.  {s}" for i, s in enumerate(SECTIONS, 1)]
TOTAL = len(SECTIONS)

if "sec" not in st.session_state:
    st.session_state.sec = LABELS[0]

def go(delta):
    i = LABELS.index(st.session_state.sec)
    st.session_state.sec = LABELS[max(0, min(TOTAL - 1, i + delta))]

with st.sidebar:
    html("""
    <div class="brand">
      <span class="brand-pill">SEAS 8525</span>
      <div class="brand-title">BERT Walkthrough</div>
      <div class="brand-course">Computer Vision &amp; Generative AI</div>
      <div class="brand-inst">Dr. Elbasheer</div>
    </div>""")

    st.radio("Sections", LABELS, key="sec", label_visibility="collapsed")
    idx = LABELS.index(st.session_state.sec)

    html(f"""
    <div class="side-label">Progress</div>
    <div class="prog-text">Section {idx + 1} of {TOTAL}</div>
    <div class="prog"><div style="width:{(idx + 1) / TOTAL * 100:.0f}%;"></div></div>
    <div class="side-label">Quick setup</div>
    <span class="setup-cmd">pip install -r requirements.txt</span>
    <span class="setup-cmd">streamlit run bert_app.py</span>
    """)

num = idx + 1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: THE BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
if num == 1:
    section_header(1, TOTAL, "The Bridge: Transformer to BERT",
                   "You already built the encoder. BERT is a stack of them, trained in a new way.")

    key_idea("BERT is the <b>Transformer encoder</b> you already know, stacked "
             "<b>12 times</b> and pretrained to fill in hidden words using context "
             "from <b>both sides</b>.")

    lead("In class you traced <b>\"the cat sat on the mat\"</b> through a Transformer encoder. "
         "Every word attended to every other word, so the output for <b>\"sat\"</b> knew about "
         "\"cat\" and \"mat\" at the same time. That two-way understanding is the foundation of BERT.")

    html(f"""
    <div class="flow">
      <div class="flow-box" style="--c:{BLUE};">
        <div class="flow-kicker">What you already know</div>
        <div class="flow-title">Transformer encoder</div>
        <ul><li>Multi-head self-attention</li><li>Feed-forward network</li>
        <li>LayerNorm and residuals</li><li>One rich vector per word</li></ul>
      </div>
      <div class="flow-op">+</div>
      <div class="flow-box" style="--c:{ORANGE};">
        <div class="flow-kicker">What BERT adds</div>
        <div class="flow-title">Stacking and pretraining</div>
        <ul><li>12 encoder blocks (24 in Large)</li><li>768-dim hidden size</li>
        <li>Special tokens [CLS] [SEP] [MASK]</li><li>Self-supervised pretraining</li></ul>
      </div>
      <div class="flow-op">=</div>
      <div class="flow-box" style="--c:{GREEN};">
        <div class="flow-kicker">The result</div>
        <div class="flow-title">A reusable language model</div>
        <ul><li>110M pretrained parameters</li><li>Fine-tune for almost any task</li>
        <li>Reads context in both directions</li><li>State of the art in 2018</li></ul>
      </div>
    </div>""")

    sub("What changes, what stays the same")
    table(
        ["Aspect", "Original Transformer", "BERT"],
        [
            ["Architecture", "Encoder and decoder stacks (6 layers each)",
             f"Encoder only, stacked {tag('12 Base', BLUE)}{tag('24 Large', BLUE)}"],
            ["Attention direction", "Encoder: both directions. Decoder: left to right",
             f"{tag('Always bidirectional', GREEN)} every token sees every token"],
            ["Hidden size", "512", "768 (Base) or 1024 (Large)"],
            ["Positional encoding", "Fixed sine and cosine waves",
             "Learned position embeddings (like ViT)"],
            ["Input tokens", "Subword tokens",
             f"WordPiece tokens plus {tag('[CLS]', NAVY)}{tag('[SEP]', NAVY)}{tag('[MASK]', RED)}"],
            ["Training objective", "Supervised translation (decoder predicts next target word)",
             f"{tag('Self-supervised', ORANGE)} Masked LM + Next Sentence Prediction"],
            ["Output used for", "Generating the translated sentence",
             "[CLS] for sentence tasks, every token for word-level tasks"],
        ],
    )

    tip("BERT comes from the 2018 paper <i>BERT: Pre-training of Deep Bidirectional "
        "Transformers for Language Understanding</i> by Devlin et al. at Google. "
        "The key word is <b>bidirectional</b>. GPT reads left to right only; "
        "BERT reads the whole sentence at once.")

    with st.expander("Frequently asked questions"):
        faq([
            ("Is BERT an encoder or a decoder?",
             "Encoder only. It produces a rich vector for each input token but does not "
             "generate new text by itself. To generate text you add a decoder (Section 7)."),
            ("How is BERT different from GPT?",
             "GPT is decoder-only and trained to predict the next token, left to right. "
             "BERT is encoder-only and trained to predict hidden tokens using both sides. "
             "GPT generates text; BERT understands it."),
            ("Why pretrain? Can we just train BERT on our task?",
             "BERT-Base has 110 million parameters. Training that from scratch needs billions "
             "of words and days of compute. Pretraining on Wikipedia and BooksCorpus gives "
             "general language understanding once; fine-tuning on your task then takes hours."),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BERT'S INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif num == 2:
    section_header(2, TOTAL, "BERT's Input: Three Embeddings",
                   "What each token looks like before the first encoder layer runs")

    key_idea("Every token enters BERT as the <b>sum of three vectors</b>: "
             "<b>what</b> the word is, <b>which sentence</b> it belongs to, "
             "and <b>where</b> it sits.")

    lead("An encoder layer only sees numbers, so each token must become a vector first. "
         "One vector per token is not enough, though. BERT has to know three separate "
         "things about every token, and it keeps <b>one learned lookup table</b> for each.")

    # ── The three questions ──────────────────────────────────────────────────
    sub("Three questions BERT asks about every token")
    grid([
        card("1. Token embedding: what is the word?",
             f"{tag('Table: 30,522 rows x 768', BLUE)}<br>"
             "<b>Think of a dictionary.</b> Each token ID picks one row of the table. "
             "The same word always gets the same row, wherever it appears.<br><br>"
             "<b>Without it:</b> BERT would not know which words it is reading.",
             BLUE),
        card("2. Segment embedding: which sentence?",
             f"{tag('Table: 2 rows x 768', PURPLE)}<br>"
             "<b>Think of team jerseys.</b> Every token in sentence A wears jersey A; "
             "every token in sentence B wears jersey B.<br><br>"
             "<b>Without it:</b> in <i>question [SEP] passage</i>, BERT could not tell "
             "which words belong to the question.",
             PURPLE),
        card("3. Position embedding: where is it?",
             f"{tag('Table: 512 rows x 768', ORANGE)}<br>"
             "<b>Think of seat numbers.</b> Position 0 gets row 0, position 1 gets row 1, "
             "and so on up to 511.<br><br>"
             "<b>Without it:</b> self-attention ignores word order, so "
             "<i>dog bites man</i> and <i>man bites dog</i> would look identical.",
             ORANGE),
    ], cols=3)

    # ── Worked example, column by column ─────────────────────────────────────
    sub("Watch it happen on one input")
    hint("Two sentences packed into one input: <b>the cat sat on the mat</b> and "
         "<b>it slept</b>. Read each column top to bottom.")

    ex_tokens = ["[CLS]", "the", "cat", "sat", "on", "the", "mat", "[SEP]", "it", "slept", "[SEP]"]
    ex_segs   = [0] * 8 + [1] * 3
    same      = {1, 5}   # the two "the" tokens

    def chip(text, color, extra=""):
        return f'<div class="chip" style="--c:{color};{extra}">{text}</div>'

    def emb_name(t):
        return {"[CLS]": "CLS", "[SEP]": "SEP"}.get(t, t)

    cols = len(ex_tokens)
    cells = '<div class="eg-label"></div>'
    for i, (t, s) in enumerate(zip(ex_tokens, ex_segs)):
        cls = "eg-col same" if i in same else "eg-col"
        cells += f'<div class="{cls}">{tok(t, token_kind(t, s))}</div>'
    rows = [
        ("Token", BLUE,   lambda i, t, s: f"E<sub>{emb_name(t)}</sub>"),
        ("Segment", PURPLE, lambda i, t, s: f"E<sub>{'A' if s == 0 else 'B'}</sub>"),
        ("Position", ORANGE, lambda i, t, s: f"E<sub>{i}</sub>"),
    ]
    for r, (name, color, fn) in enumerate(rows):
        if r > 0:
            cells += f'<div class="eg-op">+</div>' + '<div></div>' * cols
        cells += f'<div class="eg-label" style="color:{color};">{name}</div>'
        for i, (t, s) in enumerate(zip(ex_tokens, ex_segs)):
            ring = "box-shadow:0 0 0 3px #FFC400;" if (name == "Token" and i in same) else ""
            cells += chip(fn(i, t, s), color, ring)
    cells += '<div class="eg-op">=</div>' + '<div></div>' * cols
    cells += f'<div class="eg-label" style="color:{NAVY};">Input vector</div>'
    for i in range(cols):
        cells += chip(f"x<sub>{i}</sub>", NAVY, "background:#002147;color:#fff;")

    html(f'<div class="eg-wrap"><div class="eg" style="grid-template-columns:'
         f'120px repeat({cols}, minmax(58px, 1fr));">{cells}</div></div>')

    tip("Look at the two <b>the</b> tokens (gold ring). They get the <b>exact same</b> token "
        "embedding, E<sub>the</sub>. Only their position embeddings differ "
        "(E<sub>1</sub> and E<sub>5</sub>), so they enter the encoder as two "
        "<b>different</b> vectors. Notice too that the first [SEP] belongs to sentence A.")

    # ── Why add ──────────────────────────────────────────────────────────────
    sub("Why add them instead of stacking them side by side?")
    hint("A tiny made-up example with 4 numbers per vector instead of 768:")
    toy = [("Token: cat", BLUE,     [0.2, -0.5,  0.8,  0.1]),
           ("Segment: A", PURPLE,   [0.1,  0.1, -0.1,  0.0]),
           ("Position: 2", ORANGE,  [0.0,  0.3,  0.1, -0.2]),
           ("Sum = input", NAVY,    [0.3, -0.1,  0.8, -0.1])]
    table(["Vector", "dim 1", "dim 2", "dim 3", "dim 4"],
          [[tag(n, c)] + [f"{v:+.1f}" for v in vals] for n, c, vals in toy],
          num_cols=(1, 2, 3, 4), highlight_rows=(3,))
    steps([
        "<b>Same width everywhere.</b> Adding keeps each token at 768 numbers, the width "
        "every encoder layer expects. Stacking would make it 2,304.",
        "<b>Nothing important is lost.</b> The three tables are learned together, so "
        "training spreads word, sentence and position information across the 768 numbers "
        "in ways the layers can still pull apart.",
        "<b>One more step.</b> BERT applies <b>LayerNorm</b> (and dropout) to the sum. "
        "The result is what encoder layer 1 actually receives.",
    ])

    html(f"""
    <div class="key" style="background:#fff;border:1px solid #D9E1EC;border-left:9px solid {GOLD};
         box-shadow:none;text-align:center;">
      <div class="key-label" style="color:{NAVY};">The full recipe</div>
      <div class="key-text" style="color:{NAVY};font-family:'JetBrains Mono',monospace;font-size:1.15rem;">
        x<sub>i</sub> = LayerNorm(
        <span style="color:{BLUE};">E<sub>token</sub></span> +
        <span style="color:{PURPLE};">E<sub>segment</sub></span> +
        <span style="color:{ORANGE};">E<sub>position</sub></span> )
      </div>
      <div class="hint" style="margin:10px 0 0;">
        The three tables hold about <b>23.8 million</b> learned numbers, roughly a fifth of
        BERT-Base's 110 million. The original Transformer used fixed position waves and
        had no segment table.</div>
    </div>""")

    # ── Special tokens ───────────────────────────────────────────────────────
    sub("The three special tokens")
    grid([
        card("[CLS] the summary token",
             "Always the <b>first token</b>. After 12 layers its output vector is used as "
             "a summary of the whole input for sentence-level tasks. Think of it as the "
             "note-taker for the sentence.", NAVY),
        card("[SEP] the separator",
             "Marks the <b>end of a sentence</b>. With two sentences (question and context, "
             "for example) one [SEP] follows each, so BERT can tell them apart.", "#475569"),
        card("[MASK] the blank",
             "Used <b>only in pretraining</b> to hide a word BERT must predict. It never "
             "appears during fine-tuning. The 80/10/10 rule in Section 3 exists to soften "
             "that mismatch.", RED),
    ], cols=3)

    # ── Live demo ────────────────────────────────────────────────────────────
    sub("Try it: tokenize, then look inside the real vectors")
    hint("Type a sentence, or two, and press Tokenize. Try a long or unusual word such as "
         "<b>embeddings</b> to see WordPiece split it into pieces marked with ##.")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            sent1 = st.text_input("Sentence A", value="The cat sat on the mat.")
        with c2:
            sent2 = st.text_input("Sentence B (optional)", value="")
        if st.button("Tokenize", type="primary"):
            st.session_state.tok_inputs = (sent1, sent2)

    # Results stay visible while the student picks tokens below
    if st.session_state.get("tok_inputs") == (sent1, sent2):
        tokenizer = load_tokenizer()
        enc = (tokenizer(sent1, sent2, return_tensors="pt") if sent2.strip()
               else tokenizer(sent1, return_tensors="pt"))
        tokens  = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        seg_ids = enc["token_type_ids"][0].tolist()
        ids     = enc["input_ids"][0].tolist()

        sub("Result")
        token_row(tokens, seg_ids)
        token_legend(show_b=bool(sent2.strip()))

        table(
            ["Position", "Token", "Token ID", "Segment"],
            [[str(i), f'<span class="mono">{t}</span>', str(tid),
              tag("A", BLUE) if s == 0 else tag("B", PURPLE)]
             for i, (t, tid, s) in enumerate(zip(tokens, ids, seg_ids))],
            num_cols=(0, 2),
        )

        sub("Look inside: the three vectors for one token")
        hint("These are BERT's real learned numbers, first 8 of 768. With the default "
             "sentence, pick each <b>the</b> in turn: the token row stays the same while "
             "the position row changes.")

        pick = st.selectbox("Token to inspect", list(range(len(tokens))),
                            index=min(1, len(tokens) - 1),
                            format_func=lambda i: f"position {i}:   {tokens[i]}")

        import torch
        emb = load_bert().embeddings
        with torch.no_grad():
            tv = emb.word_embeddings.weight[ids[pick]]
            sv = emb.token_type_embeddings.weight[seg_ids[pick]]
            pv = emb.position_embeddings.weight[pick]
            total  = tv + sv + pv
            normed = emb.LayerNorm(total)

        seg_name = "A" if seg_ids[pick] == 0 else "B"
        vec_rows = [
            (f"Token: {tokens[pick]}", BLUE,   tv),
            (f"Segment: {seg_name}",   PURPLE, sv),
            (f"Position: {pick}",      ORANGE, pv),
            ("Sum",                    NAVY,   total),
            ("After LayerNorm",        GREEN,  normed),
        ]
        table(["Vector"] + [f"d{j}" for j in range(8)],
              [[tag(n, c)] + [f"{v:+.3f}" for v in vec.detach().numpy()[:8]]
               for n, c, vec in vec_rows],
              num_cols=tuple(range(1, 9)), highlight_rows=(4,))
        hint("Check any column: token + segment + position equals the Sum row. "
             "LayerNorm then rescales the sum so its 768 numbers have mean 0 and "
             "standard deviation 1 before learned scaling.")

    with st.expander("Show the tokenization and embedding code"):
        st.code("""
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")

enc = tokenizer("The cat sat on the mat.", return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
# ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '.', '[SEP]']

emb = model.embeddings                        # the three lookup tables
print(emb.word_embeddings.weight.shape)       # (30522, 768)  token table
print(emb.token_type_embeddings.weight.shape) # (2, 768)      segment table
print(emb.position_embeddings.weight.shape)   # (512, 768)    position table

i   = 1                                       # the first "the"
tid = enc["input_ids"][0, i]
seg = enc["token_type_ids"][0, i]
with torch.no_grad():
    x = (emb.word_embeddings.weight[tid]
         + emb.token_type_embeddings.weight[seg]
         + emb.position_embeddings.weight[i])
    x = emb.LayerNorm(x)                      # this is what layer 1 receives
print(x.shape)                                # (768,)
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MASKED LANGUAGE MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif num == 3:
    section_header(3, TOTAL, "Pretraining Task 1: Masked Language Model",
                   "How BERT learns from plain text with no human labels")

    key_idea("Hide <b>15% of the words</b> and make BERT guess them. To guess well, "
             "it has to read the words on <b>both sides</b> of the blank.")

    html(f"""
    <div class="tok-row" style="margin:6px 0 4px;">
      {tok('[CLS]','cls')}{tok('the','a')}{tok('cat','a')}{tok('[MASK]','mask')}
      {tok('on','a')}{tok('the','a')}{tok('mat','a')}{tok('[SEP]','sep')}
      <span style="font-size:1.8rem;font-weight:800;color:{NAVY};margin:0 6px;">&#8594;</span>
      {tok('sat','out')}
    </div>
    <p class="hint">Left context <b>"the cat"</b> and right context <b>"on the mat"</b>
    together make <b style="color:{GREEN};">"sat"</b> the obvious answer.</p>
    """)

    sub("The 15% masking rule")
    steps([
        "<b>Pick 15% of the tokens</b> in each sequence. In a short sentence like "
        "\"the cat sat on the mat\", that is about one word.",
        "<b>Treat each picked token one of three ways</b> (bar below).",
        "<b>Predict the original word</b> at each picked position. The loss is computed "
        "<b>only at those positions</b>, not across the whole sentence.",
        "<b>Why not always use [MASK]?</b> [MASK] never shows up at fine-tuning time. "
        "The random and unchanged cases force BERT to build a good vector for "
        "<b>every</b> token, not just the blanks.",
    ])

    html(f"""
    <div class="split">
      <div style="width:80%;background:{RED};">80%</div>
      <div style="width:10%;background:{ORANGE};">10%</div>
      <div style="width:10%;background:#64748B;">10%</div>
    </div>
    <div class="tok-row" style="gap:6px;">
      {tag('80% replaced with [MASK]: "the cat [MASK] on the mat"', RED)}
      {tag('10% random word: "the cat apple on the mat"', ORANGE)}
      {tag('10% left unchanged: "the cat sat on the mat"', '#64748B')}
    </div>""")

    tip("Compare with the Transformer decoder you studied: it predicts the <b>next</b> word "
        "from the left side only. BERT predicts a <b>hidden</b> word using both sides. "
        "In \"the cat [MASK] on the mat\", a left-to-right model only has \"the cat\" to work with.")

    sub("Try it: let BERT fill in the blank")
    hint("Put exactly one <b>[MASK]</b> in the sentence and press Predict.")

    with st.container(border=True):
        mlm_sentence = st.text_input("Sentence with one [MASK]",
                                     value="The cat [MASK] on the mat.")
        run = st.button("Predict masked word", type="primary")

    if run:
        n_masks = mlm_sentence.count("[MASK]")
        if n_masks != 1:
            st.error(f"Please use exactly one [MASK]. This sentence has {n_masks}.")
        else:
            with st.spinner("Running BERT fill-mask..."):
                results = load_fill_mask()(mlm_sentence, top_k=8)

            words  = [r["token_str"].strip() for r in results]
            scores = [r["score"] * 100 for r in results]
            top    = words[0]

            html(f"""
            <div class="key" style="border-left-color:{GREEN};">
              <div class="key-label" style="color:#6EE7B7;">BERT's top guess</div>
              <div class="key-text">{mlm_sentence.replace('[MASK]',
                f'<b style="color:#6EE7B7;">{top}</b>')}
              &nbsp;<span style="font-size:1rem;color:#CBD5E1;">({scores[0]:.1f}% confident)</span></div>
            </div>""")

            fig, ax = plt.subplots(figsize=(9, 3.8))
            colors = [GREEN] + ["#A7E3CB"] * (len(words) - 1)
            bars = ax.barh(words[::-1], scores[::-1], color=colors[::-1], height=0.65)
            for bar, sc in zip(bars, scores[::-1]):
                ax.text(bar.get_width() + max(scores) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{sc:.1f}%", va="center", fontsize=11, color="#1E293B")
            ax.set_xlim(0, max(scores) * 1.18)
            ax.set_xlabel("Probability (%)")
            ax.tick_params(axis="y", labelsize=13)
            ax.set_title("Top 8 candidates for [MASK]", loc="left")
            style_axes(ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with st.expander("Show the fill-mask code"):
        st.code("""
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

for r in fill_mask("The cat [MASK] on the mat.", top_k=5):
    print(f"{r['token_str']:12s}  {r['score']*100:.1f}%")

# BERT uses the left context ("the cat")
# AND the right context ("on the mat") to fill the blank.
# A left-to-right model like GPT only has "the cat".
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: NEXT SENTENCE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif num == 4:
    section_header(4, TOTAL, "Pretraining Task 2: Next Sentence Prediction",
                   "How BERT learns relationships between two sentences")

    key_idea("Give BERT two sentences and ask one yes-or-no question: "
             "<b>does B really follow A?</b> The answer is read from the "
             "<b>[CLS] token</b>.")

    lead("Many tasks involve two sentences: a question and a passage, a premise and a "
         "hypothesis, two sentences that may mean the same thing. Masked LM works inside one "
         "sentence, so NSP was added to teach relationships <b>between</b> sentences.")

    sub("How NSP works")
    steps([
        "BERT reads both sentences in one sequence: "
        "<span class='mono' style='font-family:JetBrains Mono,monospace;color:#002147;'>"
        "[CLS] sentence A [SEP] sentence B [SEP]</span>",
        "Half of the training pairs are <b style='color:#0E9F6E;'>IsNext</b> (B really follows A). "
        "The other half are <b style='color:#E11D48;'>NotNext</b> (B is a random sentence).",
        "After 12 layers, the <b>[CLS] output vector</b> goes to a small classifier with "
        "two outputs: IsNext or NotNext.",
        "The NSP loss and the Masked LM loss are <b>added together</b> and trained jointly.",
    ])

    sub("Two examples")
    a_side = (tok('[CLS]', 'cls') + tok('the', 'a') + tok('cat', 'a') + tok('sat', 'a')
              + tok('on', 'a') + tok('the', 'a') + tok('mat', 'a') + tok('[SEP]', 'sep'))
    grid([
        card("IsNext &#10003;",
             f'<div class="tok-row">{a_side}{tok("it","b")}{tok("fell","b")}'
             f'{tok("asleep","b")}{tok("[SEP]","sep")}</div>'
             f'[CLS] vector &#8594; classifier &#8594; <b style="color:{GREEN};">IsNext</b>',
             GREEN),
        card("NotNext &#10007;",
             f'<div class="tok-row">{a_side}{tok("stocks","b")}{tok("fell","b")}'
             f'{tok("today","b")}{tok("[SEP]","sep")}</div>'
             f'[CLS] vector &#8594; classifier &#8594; <b style="color:{RED};">NotNext</b>',
             RED),
    ])

    tip("Later work (RoBERTa, 2019) found that dropping NSP did not hurt and sometimes "
        "helped. That is why many BERT variants in Section 8 train with Masked LM only.")

    sub("Try it: build an NSP input")
    hint("Enter two sentences to see how BERT packs them and which segment each token gets.")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            nsp_a = st.text_input("Sentence A", value="The cat sat on the mat.")
        with c2:
            nsp_b = st.text_input("Sentence B", value="It fell asleep in the sun.")
        run = st.button("Build NSP input", type="primary")

    if run:
        tokenizer = load_tokenizer()
        enc = tokenizer(nsp_a, nsp_b, return_tensors="pt")
        tokens  = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        seg_ids = enc["token_type_ids"][0].tolist()

        sub("Result")
        token_row(tokens, seg_ids)
        token_legend()
        a_n, b_n = seg_ids.count(0), seg_ids.count(1)
        html(f"""<div class="tok-row">
          {tag(f'Segment A: {a_n} tokens (with [CLS] and first [SEP])', BLUE)}
          {tag(f'Segment B: {b_n} tokens (with last [SEP])', PURPLE)}
          {tag(f'Total: {len(tokens)} tokens', NAVY)}</div>""")
        hint("After all 12 layers, the [CLS] vector at position 0 encodes the relationship "
             "between the two sentences. During pretraining it feeds a 2-output layer "
             "(IsNext / NotNext).")

    with st.expander("Show the NSP code"):
        st.code("""
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

def nsp_score(a, b):
    enc = tokenizer(a, b, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits          # shape (1, 2)
    probs = logits.softmax(-1)[0]
    return {"IsNext": probs[0].item(), "NotNext": probs[1].item()}

print(nsp_score("The cat sat on the mat.", "It fell asleep in the sun."))  # IsNext high
print(nsp_score("The cat sat on the mat.", "Stocks fell sharply today."))  # NotNext high
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: THE [CLS] TOKEN
# ══════════════════════════════════════════════════════════════════════════════
elif num == 5:
    section_header(5, TOTAL, "The [CLS] Token",
                   "Which of BERT's outputs you use depends on the question you ask")

    key_idea("BERT outputs <b>one 768-number vector per token</b>. For a whole-sentence "
             "answer, read the vector at <b>[CLS]</b>. For word-level answers, read "
             "<b>every token's</b> vector.")

    lead("[CLS] attends to every other token in all 12 layers. Pretraining (NSP) teaches it "
         "to carry sentence-level information, and fine-tuning shapes it for your task.")

    sub("Which output goes where")
    table(
        ["Task type", "Output used", "Shape", "Examples"],
        [
            ["Sentence classification", tag("[CLS] only", NAVY),
             '<span class="mono">(B, 768)</span>', "Sentiment, spam detection"],
            ["Token classification", tag("Every token", GREEN),
             '<span class="mono">(B, seq, 768)</span>', "Named entities, part-of-speech tags"],
            ["Question answering", tag("Every context token", GREEN),
             '<span class="mono">(B, seq, 768)</span>', "Predict the start and end of the answer span"],
            ["Sentence-pair tasks", tag("[CLS] of the pair", NAVY),
             '<span class="mono">(B, 768)</span>', "Paraphrase detection, inference, similarity"],
        ],
    )

    sub("Try it: run BERT and inspect its outputs")
    hint("Runs bert-base-uncased on your sentence and compares the output vectors of "
         "every pair of tokens.")

    with st.container(border=True):
        embed_sent = st.text_input("Sentence", value="The cat sat on the mat.", key="embed_sent")
        center = st.checkbox(
            "Remove the shared direction first (subtract the mean vector)", value=True,
            help="BERT's last-layer vectors all point in a similar direction, so raw "
                 "cosine similarities are high for every pair. Subtracting the mean "
                 "makes the real differences visible.")
        run = st.button("Run BERT", type="primary")

    if run:
        import torch
        with st.spinner("Running a forward pass through 12 layers..."):
            tokenizer  = load_tokenizer()
            model_bert = load_bert()
            enc = tokenizer(embed_sent, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            with torch.no_grad():
                hidden = model_bert(**enc).last_hidden_state[0]   # (seq_len, 768)

        html(f"""<div class="tok-row">
          {tag(f'Output shape: {tuple(hidden.shape)}', GREEN)}
          {tag(f'{len(tokens)} tokens x 768 numbers', NAVY)}</div>""")

        cls_vec = hidden[0].numpy()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("[CLS] mean", f"{cls_vec.mean():.3f}")
        c2.metric("[CLS] std",  f"{cls_vec.std():.3f}")
        c3.metric("[CLS] min",  f"{cls_vec.min():.2f}")
        c4.metric("[CLS] max",  f"{cls_vec.max():.2f}")

        h = hidden.numpy()
        if center:
            h = h - h.mean(axis=0, keepdims=True)
        h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        sim = h @ h.T

        off = sim[~np.eye(len(tokens), dtype=bool)]
        vmin, vmax = float(off.min()), float(off.max())
        cmap = LinearSegmentedColormap.from_list("gwu", ["#FFFFFF", "#93C5FD", BLUE, NAVY])

        n = len(tokens)
        fig, ax = plt.subplots(figsize=(max(6.5, n * 0.75), max(5.5, n * 0.65)))
        im = ax.imshow(sim, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(tokens, fontsize=12)
        if n <= 16:
            for i in range(n):
                for j in range(n):
                    v = sim[i, j]
                    frac = (v - vmin) / (vmax - vmin + 1e-8)
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if frac > 0.55 else "#1E293B")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Cosine similarity", color=MUTED)
        ax.set_title("How similar is each pair of output vectors?", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        tip("The color scale is stretched to the range in <b>this</b> sentence so differences "
            "stand out. Look for which words pair up, and where [CLS] and [SEP] sit compared "
            "with the ordinary words. Untick the checkbox to see how similar the raw vectors "
            "all are.")

    with st.expander("Show the embedding extraction code"):
        st.code("""
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")

enc = tokenizer("The cat sat on the mat.", return_tensors="pt")
with torch.no_grad():
    out = model(**enc)

# out.last_hidden_state : (batch, seq_len, 768)  last-layer output for every token
# out.pooler_output     : (batch, 768)           [CLS] passed through linear + tanh

hidden          = out.last_hidden_state[0]   # (seq_len, 768)
cls_embedding   = hidden[0]                  # (768,)  the [CLS] token
word_embeddings = hidden[1:-1]               # without [CLS] and [SEP]
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════
elif num == 6:
    section_header(6, TOTAL, "Fine-tuning for Downstream Tasks",
                   "Same pretrained BERT, a different small head for each task")

    key_idea("Keep the pretrained BERT, add <b>one small layer</b> on top, and train "
             "<b>both together</b> on your labeled data for a few epochs.")

    lead("Because BERT already understands language, fine-tuning usually needs only "
         "<b>2 to 4 epochs</b> on a few thousand labeled examples.")

    sub("Four fine-tuning patterns")
    grid([
        card("A. Sentence classification",
             f"{tag('reads [CLS]', NAVY)}<br>One sentence in, one label out.<br>"
             "<b>Tasks:</b> sentiment, spam, topic.<br>"
             "<b>Head:</b> <code>nn.Linear(768, num_classes)</code>", BLUE),
        card("B. Token classification",
             f"{tag('reads every token', GREEN)}<br>One label per token.<br>"
             "<b>Tasks:</b> named entities, part-of-speech tags.<br>"
             "<b>Head:</b> <code>nn.Linear(768, num_labels)</code> on each token", GREEN),
        card("C. Question answering",
             f"{tag('reads every context token', GREEN)}<br>"
             "Input: [CLS] question [SEP] passage [SEP]. Two scores per token: "
             "answer <b>start</b> and answer <b>end</b>.<br>"
             "<b>Head:</b> <code>nn.Linear(768, 2)</code> on each token", ORANGE),
        card("D. Sentence-pair tasks",
             f"{tag('reads [CLS]', NAVY)}<br>"
             "Input: [CLS] A [SEP] B [SEP]. One label for the relationship.<br>"
             "<b>Tasks:</b> inference, paraphrase, similarity.<br>"
             "<b>Head:</b> <code>nn.Linear(768, num_classes)</code>", PURPLE),
    ])

    tip("In all four patterns the <b>whole model is trained</b>, not just the head. "
        "A small learning rate (2e-5 to 5e-5) keeps the pretrained weights from "
        "changing too much.")

    sub("Try it: sentiment analysis with a fine-tuned model")
    hint("This demo uses DistilBERT (a smaller BERT, see Section 8) fine-tuned on SST-2, "
         "the Stanford Sentiment Treebank. It is pattern A in action.")

    with st.container(border=True):
        sent_input = st.text_area("Text to analyze",
                                  value="The cat sat on the mat and looked very content.",
                                  height=90)
        run = st.button("Analyze sentiment", type="primary")

    if run:
        with st.spinner("Running sentiment analysis..."):
            result = load_sentiment()(sent_input)[0]

        label, score = result["label"], result["score"]
        color = GREEN if label == "POSITIVE" else RED
        html(f"""
        <div class="key" style="border-left-color:{color};">
          <div class="key-label" style="color:{'#6EE7B7' if label == 'POSITIVE' else '#FDA4AF'};">Prediction</div>
          <div class="key-text">{label.title()}
          <span style="font-size:1.05rem;color:#CBD5E1;">&nbsp;{score * 100:.1f}% confident</span></div>
        </div>""")

        pos = score if label == "POSITIVE" else 1 - score
        fig, ax = plt.subplots(figsize=(9, 1.4))
        ax.barh([0], [pos], color=GREEN, height=0.6)
        ax.barh([0], [1 - pos], left=[pos], color=RED, height=0.6)
        if pos > 0.08:
            ax.text(pos / 2, 0, f"Positive {pos * 100:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold")
        if 1 - pos > 0.08:
            ax.text(pos + (1 - pos) / 2, 0, f"Negative {(1 - pos) * 100:.0f}%",
                    ha="center", va="center", color="white", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("Show the fine-tuning code"):
        st.code("""
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)      # positive / negative

texts  = ["I love this movie!", "This film was terrible."]
labels = torch.tensor([1, 0])

enc = tokenizer(texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt")

outputs = model(**enc, labels=labels)
loss    = outputs.loss      # cross-entropy on the [CLS] head
logits  = outputs.logits    # (batch, 2)

# Small learning rate: the pretrained weights should move only a little
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss.backward()
optimizer.step()
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ENCODER TO DECODER
# ══════════════════════════════════════════════════════════════════════════════
elif num == 7:
    section_header(7, TOTAL, "Sending Encoder Output to a Decoder",
                   "When you need to generate text, cross-attention connects the two")

    key_idea("BERT <b>understands</b> text but cannot <b>write</b> it. To generate, hand its "
             "output to a <b>decoder</b>, which reads it through <b>cross-attention</b>.")

    sub("Two ways to use BERT's output")
    grid([
        card("Path A: a task head (Section 6)",
             "Feed BERT's outputs to a small linear layer. No decoder. The output has a "
             "fixed size: a class, a label per token, or a start and end position.<br>"
             f"{tag('Classification', BLUE)}{tag('Named entities', BLUE)}{tag('Span QA', BLUE)}",
             BLUE),
        card("Path B: a decoder (this section)",
             "Pass the full <b>(src_len, 768)</b> output matrix to a Transformer decoder. "
             "The decoder writes one token at a time, looking back at the encoder output "
             "through cross-attention.<br>"
             f"{tag('Translation', GREEN)}{tag('Summarization', GREEN)}{tag('Generation', GREEN)}",
             GREEN),
    ])

    sub("How cross-attention works")
    lead("In self-attention, Q, K and V all come from the same sequence. In cross-attention, "
         "<b>Q comes from the decoder</b> and <b>K and V come from the encoder</b>.")

    html(f"""
    <div class="xattn">
      <div class="x-box" style="--c:{BLUE};">
        <div class="x-title">Encoder output</div>
        <div class="x-desc">(src_len, 768)<br>becomes <b>K</b> and <b>V</b></div>
      </div>
      <div class="arrow">&#8594;</div>
      <div class="x-core">
        <div style="font-weight:800;font-size:1.1rem;">Cross-attention</div>
        <div class="f">softmax(Q K<sup>T</sup> / &#8730;d) V</div>
        <div style="font-size:.9rem;color:#CBD5E1;margin-top:6px;">
          each decoder position looks at every source position</div>
      </div>
      <div class="arrow">&#8592;</div>
      <div class="x-box" style="--c:{GREEN};">
        <div class="x-title">Decoder state</div>
        <div class="x-desc">(tgt_len, 768)<br>becomes <b>Q</b></div>
      </div>
    </div>""")

    steps([
        "The <b>encoder</b> reads the whole source sentence once and produces its output matrix.",
        "The <b>decoder</b> writes the output one token at a time. Its current state makes the "
        "<b>queries Q</b>.",
        "<b>Cross-attention</b> scores each query against every encoder position (the keys K) "
        "and mixes the matching values V.",
        "Each decoder layer has three parts: <b>masked self-attention</b> on the words written "
        "so far, <b>cross-attention</b> to the encoder, and a <b>feed-forward</b> network.",
    ])

    sub("The full picture")
    table(
        ["Stage", "Encoder", "Decoder"],
        [
            ["Input", "Source tokens (e.g. an English sentence)",
             "Target tokens written so far (e.g. French)"],
            ["Attention", tag("Self-attention, bidirectional", BLUE),
             tag("Masked self-attention, left to right", GREEN) + "<br>"
             + tag("Cross-attention to encoder", ORANGE)],
            ["Output", "(src_len, 768) matrix handed to the decoder",
             "Probabilities over the vocabulary for the next token"],
            ["Examples", "T5 encoder, BART encoder, or a pretrained BERT",
             "T5 decoder, BART decoder"],
        ],
    )

    tip("T5 and BART use this same encoder-decoder design but train their own encoders. "
        "You can also plug a pretrained BERT in as the encoder, for example with Hugging Face's "
        "<code>EncoderDecoderModel</code> (\"BERT2BERT\").")

    with st.expander("Show cross-attention in PyTorch"):
        st.code("""
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads          # 64
        # Q comes from the decoder; K and V come from the encoder
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, decoder_hidden, encoder_output):
        # decoder_hidden: (B, tgt_len, 768)   source of Q
        # encoder_output: (B, src_len, 768)   source of K and V
        B, T, D = decoder_hidden.shape
        S = encoder_output.shape[1]

        Q = self.W_q(decoder_hidden).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(encoder_output).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(encoder_output).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # every decoder position attends to ALL encoder positions
        attn = (Q @ K.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(-1)                              # (B, H, T, S)

        out = (attn @ V).transpose(1, 2).reshape(B, T, D)    # (B, T, 768)
        return self.W_o(out)
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: BERT VARIANTS
# ══════════════════════════════════════════════════════════════════════════════
elif num == 8:
    section_header(8, TOTAL, "BERT Variants",
                   "How the community built on BERT after 2018")

    key_idea("Most variants keep BERT's architecture and change <b>how it is trained</b>: "
             "more data, different tasks, fewer parameters, or a specific domain.")

    sub("The family at a glance")
    table(
        ["Model", "Params", "Layers", "Hidden", "Heads", "What changed from BERT-Base"],
        [
            ["BERT-Base", "110M", "12", "768", "12",
             f"{tag('The original', NAVY)} Masked LM + NSP on Wikipedia and BooksCorpus "
             "(3.3B words)"],
            ["BERT-Large", "340M", "24", "1024", "16",
             f"{tag('Bigger', BLUE)} Twice as deep and wider; about 3x the parameters"],
            ["RoBERTa", "125M", "12", "768", "12",
             f"{tag('Better training', ORANGE)} No NSP, 10x more data, a new mask each "
             "epoch, much larger batches"],
            ["DistilBERT", "66M", "6", "768", "12",
             f"{tag('Smaller', GREEN)} Distilled from BERT-Base: 40% smaller, 60% faster, "
             "keeps about 97% of its accuracy"],
            ["ALBERT-base", "12M", "12", "768", "12",
             f"{tag('Shared weights', GREEN)} One set of layer weights reused 12 times; "
             "sentence order prediction replaces NSP"],
            ["BioBERT", "110M", "12", "768", "12",
             f"{tag('Biomedical', PURPLE)} BERT-Base further pretrained on PubMed and PMC "
             "articles"],
            ["SciBERT", "110M", "12", "768", "12",
             f"{tag('Scientific', PURPLE)} Trained from scratch on 1.14M papers with its own "
             "science vocabulary"],
        ],
        num_cols=(1, 2, 3, 4),
        highlight_rows=(0,),
    )

    sub("Which one should you use?")
    grid([
        card("General NLP tasks",
             "Start with <b>RoBERTa-base</b>. Same size as BERT-Base and usually better. "
             "Need something smaller? Use <b>DistilBERT</b>.", GREEN),
        card("Domain-specific text",
             "Biomedical: <b>BioBERT</b> or <b>PubMedBERT</b>. Legal: <b>LegalBERT</b>. "
             "Science: <b>SciBERT</b>. Domain pretraining often gives a clear boost "
             "on specialized tasks.", PURPLE),
        card("Production or limited hardware",
             "<b>DistilBERT</b> or <b>ALBERT-base</b> for speed and memory. Both run "
             "comfortably on a CPU.", BLUE),
        card("Maximum accuracy",
             "<b>RoBERTa-large</b> or <b>DeBERTa-large</b>. DeBERTa's disentangled attention "
             "beats RoBERTa on GLUE and SuperGLUE. Needs a large GPU to fine-tune.", ORANGE),
    ])

    with st.expander("Show how to load any variant from Hugging Face"):
        st.code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Change only this line to switch variants
MODEL = "roberta-base"    # or "distilbert-base-uncased", "allenai/scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# AutoTokenizer and AutoModel detect the architecture from the model card,
# so you never have to pick BertTokenizer vs RobertaTokenizer yourself.
enc = tokenizer("The cat sat on the mat.", return_tensors="pt",
                truncation=True, max_length=512)
print(model(**enc).logits.shape)   # (1, 2)
""", language="python")


# ── Previous / Next ───────────────────────────────────────────────────────────
st.divider()
p, _, n = st.columns([1, 2, 1])
with p:
    st.button("←  Previous", on_click=go, args=(-1,), disabled=num == 1,
              use_container_width=True, key="prev_btn")
with n:
    st.button("Next  →", on_click=go, args=(1,), disabled=num == TOTAL,
              type="primary", use_container_width=True, key="next_btn")
