import io
import base64
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import timm
from torchvision import transforms


st.set_page_config(
    page_title="WildID",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_ROOT = PROJECT_ROOT / "R001_Demo"
IMAGE_DIR = DEMO_ROOT / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
GALLERY_FILE = RESULTS_DIR / "gallery_embeddings.pt"

MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-T-224"
THRESHOLD = 0.90
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD_LABEL = f"{THRESHOLD:.2f}"

@st.cache_resource(show_spinner="Loading WildID model...")
def load_model():
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,
    )
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_gallery():
    data = torch.load(GALLERY_FILE, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    identities = np.asarray(data["identities"])
    filenames = list(data["filenames"])
    return embeddings, identities, filenames


@st.cache_data(show_spinner=False)
def get_demo_image(filename):
    path = IMAGE_DIR / filename
    return Image.open(path).convert("RGB")


@st.cache_data(show_spinner=False)
def get_hero_image_data():
    path = IMAGE_DIR / "002253.jpg"
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def embed_image(image, model):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        embedding = model(tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=1)
    return embedding.cpu()


def search_gallery(image, model, gallery_embeddings, gallery_ids, gallery_filenames):
    query_embedding = embed_image(image, model)
    scores = (query_embedding @ gallery_embeddings.T).squeeze(0).numpy()
    ranking = np.argsort(-scores)

    top = ranking[:5]
    top1_score = float(scores[top[0]])
    top1_identity = int(gallery_ids[top[0]])

    decision = "MATCH" if top1_score >= THRESHOLD else "NO RELIABLE MATCH"

    candidates = []
    for rank, idx in enumerate(top, start=1):
        candidates.append(
            {
                "rank": rank,
                "identity": int(gallery_ids[idx]),
                "filename": gallery_filenames[idx],
                "similarity": float(scores[idx]),
            }
        )

    return {
        "decision": decision,
        "top1_score": top1_score,
        "top1_identity": top1_identity,
        "candidates": candidates,
    }
    

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&family=Playfair+Display:ital,wght@0,400;0,500;1,400;1,500&display=swap');

:root {
    --bg: #0b0d0b;
    --surface: #111410;
    --surface-2: #151915;
    --line: #2b302b;
    --text: #eee9dc;
    --muted: #a6a99f;
    --dim: #8a9086;
    --warm: #d9c9a9;
    --green: #829276;
    --green-bright: #a2b494;
    --amber: #c9ad78;

    --maxw: 1240px;
    --gutter: clamp(1.25rem, 5vw, 3.6rem);
    --pad-x: max(var(--gutter), calc((100% - var(--maxw)) / 2));
    --section-gap: clamp(2.5rem, 5vw, 4.25rem);

    --mono: "DM Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    --sans: "DM Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    --serif: "Playfair Display", Georgia, "Times New Roman", serif;
}

/* ---- Streamlit shell ---------------------------------------------------- */

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
}

[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
}
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
#MainMenu, footer { display: none !important; }

.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: none !important;
    padding: 0 !important;
    overflow-x: clip;
}

/* Remove Streamlit's default 1rem gap between top-level elements; spacing is
   controlled explicitly by the page containers below. */
.block-container > [data-testid="stVerticalBlock"],
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

[data-testid="stMarkdownContainer"] * { box-sizing: border-box; }

/* Page body container (st.container(key="wid-...")): shared max width, padding
   and alignment with the header and hero. */
[class*="st-key-wid-"] {
    width: 100%;
    gap: 0 !important;
    padding: clamp(2rem, 4vw, 3.25rem) var(--pad-x) clamp(3rem, 6vw, 5rem) !important;
}
.st-key-wid-overview {
    padding-top: clamp(2rem, 4vw, 3rem) !important;
}
[class*="st-key-wid-"] [data-testid="stColumn"] { min-width: 0; }

/* ---- Header -------------------------------------------------------------- */

.wildid-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    min-height: 78px;
    padding: 0 var(--pad-x);
    background: #090b09;
    border-bottom: 1px solid rgba(238, 232, 218, 0.14);
}

.wildid-brand {
    display: flex;
    align-items: baseline;
    gap: 1.1rem;
    flex: 1 1 0;
    min-width: 0;
    text-decoration: none !important;
}

.wildid-brand-name {
    font-family: var(--mono);
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.34em;
    color: #f4f0e6 !important;
    -webkit-text-fill-color: #f4f0e6 !important;
    white-space: nowrap;
}

.wildid-brand-sub {
    color: #d1d0c8 !important;
    -webkit-text-fill-color: #d1d0c8 !important;
    font-family: var(--sans);
    font-size: 0.72rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.wildid-nav {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: clamp(1.1rem, 2vw, 2.4rem);
    flex: 0 0 auto;
}

.wildid-nav a {
    color: #eee9dc !important;
    -webkit-text-fill-color: #eee9dc !important;
    text-decoration: none !important;
    font-family: var(--sans);
    font-size: 0.78rem;
    font-weight: 500;
    line-height: 1;
    white-space: nowrap;
    padding: 0.8rem 0.2rem 0.72rem;
    border-bottom: 1px solid transparent;
    transition: color 120ms ease, border-color 120ms ease;
}

.wildid-nav a:hover,
.wildid-nav a[aria-current="page"] {
    color: #fffdf6 !important;
    -webkit-text-fill-color: #fffdf6 !important;
    border-bottom-color: var(--warm);
}

.wildid-nav a:focus-visible {
    outline: 2px solid var(--warm);
    outline-offset: 3px;
}

.wildid-status {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 1rem;
    flex: 1 1 0;
    min-width: 0;
}

.wildid-prototype {
    border: 1px solid #363a34;
    padding: 0.52rem 0.72rem;
    color: #d9cdb4 !important;
    -webkit-text-fill-color: #d9cdb4 !important;
    font-family: var(--mono);
    font-size: 0.61rem;
    letter-spacing: 0.14em;
    white-space: nowrap;
}

.wildid-baseline {
    color: #cfc5b1 !important;
    -webkit-text-fill-color: #cfc5b1 !important;
    font-family: var(--mono);
    font-size: 0.61rem;
    letter-spacing: 0.14em;
    white-space: nowrap;
}

/* ---- Typography ---------------------------------------------------------- */

.eyebrow {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: var(--dim);
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

.page-head { margin-bottom: clamp(1.75rem, 3vw, 2.5rem); }

.page-title {
    font-family: var(--serif);
    font-weight: 400;
    color: var(--text);
    font-size: clamp(2.2rem, 4.6vw, 3.8rem);
    line-height: 1.04;
    letter-spacing: -0.03em;
    max-width: 900px;
    margin: 0;
    overflow-wrap: break-word;
    text-wrap: balance;
}

.lede {
    max-width: 720px;
    color: #a9ada4;
    font-family: var(--sans);
    font-size: 1.02rem;
    line-height: 1.65;
    margin-top: 1rem;
    overflow-wrap: break-word;
}
.lede strong { color: var(--text); font-weight: 500; }

.section-label,
.panel-title {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    color: var(--dim);
    text-transform: uppercase;
}
.panel-title { margin-bottom: 1.1rem; }
.section-label { margin-top: 0.4rem; }

.h-section {
    font-family: var(--serif);
    font-weight: 400;
    color: var(--text);
    font-size: clamp(1.8rem, 3.2vw, 2.8rem);
    line-height: 1.08;
    letter-spacing: -0.02em;
    margin: 0.5rem 0 1.1rem;
    text-wrap: balance;
}

.h-card {
    font-family: var(--serif);
    font-weight: 400;
    color: var(--text);
    font-size: clamp(1.5rem, 2.4vw, 2.1rem);
    line-height: 1.12;
    letter-spacing: -0.015em;
    margin: 0.4rem 0 0.9rem;
}

.copy {
    color: #9da198;
    font-family: var(--sans);
    font-size: 0.96rem;
    line-height: 1.75;
    max-width: 720px;
    overflow-wrap: break-word;
}
.copy + .copy { margin-top: 1rem; }
.copy strong { color: var(--warm); font-weight: 600; }

.fineprint {
    color: var(--dim);
    font-size: 0.78rem;
    line-height: 1.55;
    max-width: 820px;
}

.mt-section { margin-top: var(--section-gap); }
.mt-block { margin-top: 1.25rem; }

/* ---- Hero ---------------------------------------------------------------- */

.hero-wrap {
    position: relative;
    display: flex;
    align-items: center;
    width: 100%;
    min-height: clamp(540px, calc(100vh - 150px), 760px);
    overflow: hidden;
    background: #0b0d0b;
}

.hero-image {
    position: absolute;
    inset: 0;
    background-color: #0b0d0b;
    background-image: url("https://images.unsplash.com/photo-1698382439843-ca033a6079c0?fm=jpg&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&ixlib=rb-4.1.0&q=85&w=2600");
    background-size: cover;
    background-position: center 42%;
    filter: saturate(0.46) contrast(0.95) brightness(0.58);
}

.hero-overlay {
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(7,9,7,0.88) 0%, rgba(7,9,7,0.63) 31%, rgba(7,9,7,0.18) 70%, rgba(7,9,7,0.32) 100%),
        linear-gradient(180deg, rgba(7,9,7,0.38) 0%, rgba(7,9,7,0.02) 34%, rgba(7,9,7,0.18) 64%, rgba(7,9,7,0.76) 91%, #0b0d0b 100%);
}

.hero-content {
    position: relative;
    z-index: 2;
    width: 100%;
    padding: 4.5rem var(--pad-x) 5rem;
}

.hero-eyebrow { color: #d6c7a9 !important; margin-bottom: 1.05rem; }

.hero-title {
    font-family: var(--serif);
    font-weight: 400;
    font-size: clamp(2.5rem, 6vw, 5.6rem);
    line-height: 0.96;
    letter-spacing: -0.045em;
    color: #f2eee5;
    margin: 0;
    overflow-wrap: break-word;
}
.hero-title em { color: #d8c6a5; font-style: italic; }

.hero-lede {
    max-width: 700px;
    color: #c3c4bc;
    font-size: 1.02rem;
    line-height: 1.6;
    margin-top: 1.5rem;
}

.hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 2rem;
}

.hero-action-primary,
.hero-action-secondary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 2.9rem;
    border: 1px solid #45443c;
    padding: 0.78rem 1.35rem;
    font-family: var(--sans);
    font-size: 0.8rem;
    text-decoration: none !important;
    transition: background 120ms ease, border-color 120ms ease;
}

.hero-action-primary,
.hero-action-primary:visited {
    color: #0b0d0b !important;
    -webkit-text-fill-color: #0b0d0b !important;
    background: #e0d2b8;
    border-color: #e0d2b8;
}
.hero-action-primary:hover { background: #eee2c8; }

.hero-action-secondary,
.hero-action-secondary:visited {
    color: #dedbd1 !important;
    -webkit-text-fill-color: #dedbd1 !important;
    background: rgba(10, 12, 10, 0.38);
    border-color: #5a5a50;
}
.hero-action-secondary:hover { border-color: var(--warm); }

.hero-action-primary:focus-visible,
.hero-action-secondary:focus-visible {
    outline: 2px solid var(--warm);
    outline-offset: 3px;
}

/* ---- Panels, steps, metrics --------------------------------------------- */

.panel {
    border: 1px solid var(--line);
    background: var(--surface);
    padding: 1.5rem clamp(1.1rem, 2.5vw, 1.6rem);
}

.step-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    scroll-margin-top: 1.5rem;
}

.step {
    padding: 1.5rem clamp(1.1rem, 2.5vw, 2rem) 1.75rem;
    border-right: 1px solid var(--line);
}
.step:last-child { border-right: none; }

.step-no {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: #747a70;
    margin-bottom: 1.25rem;
}

.step-title {
    font-family: var(--serif);
    font-size: clamp(1.4rem, 2.2vw, 1.7rem);
    color: var(--text);
    margin-bottom: 0.6rem;
}

.step-copy { color: #92978e; line-height: 1.55; font-size: 0.95rem; }

.evidence-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr);
    gap: 1.25rem;
    align-items: stretch;
}

.kv-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
    padding: 0.8rem 0;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.95rem;
}
.kv-row strong {
    color: var(--text);
    font-weight: 500;
    text-align: right;
    overflow-wrap: anywhere;
}
.kv-plain .kv-row:first-of-type { border-top: none; padding-top: 0.2rem; }

.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    border: 1px solid var(--line);
    background: var(--surface);
    margin-bottom: 1rem;
}

.metric {
    padding: 1.1rem clamp(0.9rem, 2vw, 1.4rem) 1.2rem;
    border-right: 1px solid var(--line);
}
.metric:last-child { border-right: none; }

.metric-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    color: #747a70;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-family: var(--serif);
    font-size: clamp(1.6rem, 3vw, 2rem);
    color: var(--text);
    line-height: 1.1;
}

.panel .fineprint { margin-top: 1rem; }

/* ---- Identify ------------------------------------------------------------ */

[data-testid="stFileUploader"] { color: var(--text); }

[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed #454940 !important;
    background: #0e110e !important;
    border-radius: 0 !important;
}

[data-testid="stFileUploader"] label p,
[data-testid="stWidgetLabel"] p {
    color: #dedbd2 !important;
    font-family: var(--sans);
}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--muted) !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #ded0b2 !important;
    color: #0a0c0a !important;
    border: 0 !important;
    border-radius: 0 !important;
}

[data-testid="stFileUploaderFileName"] {
    color: var(--text) !important;
    overflow-wrap: anywhere;
}

.stButton > button {
    width: 100%;
    min-height: 2.75rem;
    border: 1px solid #4a4f46;
    border-radius: 0;
    background: transparent;
    color: var(--text);
    font-family: var(--sans);
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.6rem 1rem;
    transition: background 120ms ease, border-color 120ms ease;
}
.stButton > button p { color: inherit !important; font-size: inherit; }
.stButton > button:hover {
    border-color: var(--warm);
    background: rgba(217, 201, 169, 0.09);
    color: #fffdf6;
}
.stButton > button:focus-visible {
    outline: 2px solid var(--warm) !important;
    outline-offset: 2px;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: #d9c9a9;
    border-color: #d9c9a9;
    color: #0b0d0b;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #eee2c8;
    border-color: #eee2c8;
    color: #0b0d0b;
}

[data-testid="stImage"] img {
    border-radius: 0 !important;
    border: 1px solid var(--line);
    background: #0a0c0a;
    max-height: 440px;
    object-fit: contain;
}

img { border-radius: 0 !important; }

.img-caption {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.35rem 0.9rem;
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--dim);
}
.img-caption b {
    color: var(--text);
    font-weight: 500;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.notice {
    border: 1px solid #4b4334;
    background: #14120e;
    color: #b8b2a3;
    font-size: 0.82rem;
    line-height: 1.5;
    padding: 0.7rem 0.9rem;
}

.panel-heading {
    font-family: var(--serif);
    font-size: clamp(1.5rem, 2.6vw, 2rem);
    color: var(--text);
    margin-bottom: 0.6rem;
}

.list-rows { margin-top: 1.1rem; }
.list-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.25rem 1rem;
    padding: 0.85rem 0;
    border-top: 1px solid var(--line);
    color: var(--text);
}
.list-row em {
    font-style: normal;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: #747a70;
    margin-right: 0.8rem;
}
.list-row small { color: var(--dim); font-size: 0.85rem; }

.result-meta {
    font-family: var(--mono);
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    color: var(--dim);
    text-transform: uppercase;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.result-meta b { color: var(--muted); font-weight: 500; text-transform: none; letter-spacing: 0.04em; }

.result-card {
    border: 1px solid var(--line);
    border-left-width: 3px;
    padding: 1.25rem 1.4rem;
}
.result-card.is-match { border-left-color: var(--green); background: #111811; }
.result-card.is-no { border-left-color: var(--amber); background: #17140e; }

.result-state {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    margin-bottom: 0.6rem;
}
.is-match .result-state { color: var(--green-bright); }
.is-no .result-state { color: var(--amber); }

.result-identity {
    font-family: var(--serif);
    font-size: clamp(1.8rem, 3vw, 2.4rem);
    line-height: 1.1;
    color: var(--text);
    overflow-wrap: anywhere;
}

.result-score {
    margin-top: 0.55rem;
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--muted);
}
.result-score b { font-size: 1.15rem; font-weight: 500; color: var(--warm); margin-right: 0.4rem; }

.result-card .fineprint { margin-top: 0.7rem; }

.cand-panel {
    border: 1px solid var(--line);
    background: var(--surface);
    container-type: inline-size;
}
.cand-title { padding: 1.1rem 1rem 0.9rem; }
.cand-title .panel-title { margin-bottom: 0; }

.cand-row {
    display: grid;
    grid-template-columns: 2rem minmax(0, 1.1fr) minmax(0, 1.4fr) 4.4rem;
    grid-template-areas: "rank id bar score";
    align-items: center;
    column-gap: 1rem;
    padding: 0.85rem 1rem;
    border-top: 1px solid var(--line);
}
.cand-row.is-top {
    background: rgba(217, 201, 169, 0.05);
    box-shadow: inset 3px 0 0 var(--warm);
}

.cand-rank { grid-area: rank; font-family: var(--mono); font-size: 0.72rem; color: #72786e; }
.cand-id { grid-area: id; min-width: 0; }
.cand-id-main { display: block; font-family: var(--mono); font-size: 0.78rem; color: var(--text); }
.cand-file {
    display: block;
    margin-top: 0.15rem;
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.cand-bar { grid-area: bar; position: relative; height: 3px; background: #2a2f2a; }
.cand-bar-fill { position: absolute; top: 0; bottom: 0; left: 0; background: #59634f; }
.is-top .cand-bar-fill { background: var(--bar-top, var(--green)); }
.cand-bar-tick {
    position: absolute;
    left: 90%;
    top: -4px;
    bottom: -4px;
    width: 1px;
    background: var(--warm);
    opacity: 0.75;
}
.cand-score {
    grid-area: score;
    text-align: right;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--warm);
}

.cand-legend {
    padding: 0.8rem 1rem;
    border-top: 1px solid var(--line);
    font-size: 0.74rem;
    color: var(--dim);
    line-height: 1.5;
}

@container (max-width: 460px) {
    .cand-row {
        grid-template-columns: 2rem minmax(0, 1fr) auto;
        grid-template-areas: "rank id score" ". bar bar";
        row-gap: 0.6rem;
    }
}

/* ---- Individuals / Sightings -------------------------------------------- */

.status-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
    gap: 1.25rem;
    align-items: stretch;
}

.tag {
    display: inline-block;
    border: 1px solid #4b4334;
    color: var(--warm);
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.13em;
    padding: 0.3rem 0.55rem;
    white-space: nowrap;
}
.tag.is-ok { border-color: #3b4a35; color: var(--green-bright); }

.status-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem 1rem;
    padding: 0.85rem 0;
    border-top: 1px solid var(--line);
    color: var(--text);
    font-size: 0.95rem;
}
.status-row small { display: block; color: var(--dim); font-size: 0.8rem; margin-top: 0.15rem; }

.schematic-grid {
    display: grid;
    grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr);
    gap: 1rem;
}
.schematic-media {
    min-height: 150px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 1rem;
    border: 1px dashed #454940;
    color: var(--dim);
    font-family: var(--mono);
    font-size: 0.64rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.schematic-field {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.7rem 0.8rem;
    margin-bottom: 0.5rem;
    border: 1px dashed #3a3f38;
    color: var(--muted);
    font-size: 0.85rem;
}
.schematic-field:last-child { margin-bottom: 0; }
.schematic-field i { font-style: normal; color: #5f655b; }

.compare { container-type: inline-size; }
.compare-row {
    display: grid;
    grid-template-columns: 6.5rem minmax(0, 1fr) minmax(0, 1fr);
    gap: 1rem;
    padding: 0.85rem 0;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.5;
}
.compare-row.is-head {
    border-top: none;
    padding-top: 0;
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--dim);
}
.compare-key { font-family: var(--mono); font-size: 0.64rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--dim); }

@container (max-width: 520px) {
    .compare-row { grid-template-columns: minmax(0, 1fr); gap: 0.3rem; }
    .compare-row.is-head { display: none; }
}

/* ---- About --------------------------------------------------------------- */

.about-intro {
    display: grid;
    grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
    gap: clamp(2rem, 5vw, 5rem);
    align-items: stretch;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
}
.about-intro-copy { padding: clamp(2rem, 4vw, 3.5rem) 0; }
.about-intro-image {
    position: relative;
    overflow: hidden;
    background: var(--surface);
    min-height: clamp(280px, 38vw, 480px);
}
.about-intro-image::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: url("https://images.unsplash.com/photo-1715088295646-de8244579a3d?auto=format&fit=crop&fm=jpg&ixlib=rb-4.1.0&q=88&w=2200");
    background-size: cover;
    background-position: center 48%;
    filter: saturate(0.62) contrast(1.03) brightness(0.76);
}
.about-intro-image::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, #0b0d0b 0%, rgba(11,13,11,0.10) 28%, rgba(11,13,11,0.06) 100%),
        linear-gradient(180deg, rgba(11,13,11,0.02), rgba(11,13,11,0.55));
}
.about-image-label {
    position: absolute;
    z-index: 2;
    left: 1.25rem;
    bottom: 1.25rem;
    font-family: var(--mono);
    font-size: 0.64rem;
    letter-spacing: 0.16em;
    color: var(--warm);
    text-transform: uppercase;
}

.about-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
    gap: 1px;
    background: var(--line);
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
}
.about-card { background: var(--bg); padding: clamp(1.5rem, 3vw, 2.5rem); }
.about-card-wide { grid-row: span 2; }

.about-method {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: clamp(2rem, 5vw, 4rem);
    padding-bottom: var(--section-gap);
    border-bottom: 1px solid var(--line);
}

.method-list { border-top: 1px solid var(--line); align-self: start; }
.method-list > div {
    display: grid;
    grid-template-columns: 2.8rem minmax(0, 1fr) auto;
    gap: 1rem;
    align-items: center;
    padding: 1.1rem 0;
    border-bottom: 1px solid var(--line);
}
.method-list span { font-family: var(--mono); color: #6f756c; font-size: 0.7rem; }
.method-list strong { color: var(--text); font-family: var(--sans); font-weight: 500; }
.method-list small { color: var(--dim); font-family: var(--sans); font-size: 0.85rem; }

.about-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem 2rem;
    padding: 0.25rem 0 1rem;
    color: var(--dim);
    font-family: var(--mono);
    font-size: 0.68rem;
}
.about-stats b { color: var(--text); font-size: 0.9rem; font-weight: 500; }

.about-abstain {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 330px);
    gap: clamp(2rem, 5vw, 4rem);
    align-items: center;
    padding: clamp(2.5rem, 5vw, 4rem) 0;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
}
.abstain-badge { border: 1px solid #4b4334; background: #14120e; padding: 1.75rem; }
.abstain-badge span, .abstain-badge small {
    display: block;
    font-family: var(--mono);
    color: #857e6c;
    font-size: 0.64rem;
    letter-spacing: 0.13em;
}
.abstain-badge strong {
    display: block;
    color: var(--warm);
    font-family: var(--mono);
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    margin: 0.7rem 0;
}

.roadmap-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1px;
    background: var(--line);
    margin-top: 1rem;
}
.roadmap-grid > div { background: var(--bg); padding: clamp(1.4rem, 2.5vw, 2rem); }
.roadmap-grid span {
    display: block;
    font-family: var(--mono);
    color: #d6c7a9;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    margin-bottom: 1.5rem;
}
.roadmap-grid strong {
    display: block;
    color: var(--text);
    font-family: var(--serif);
    font-size: clamp(1.4rem, 2.2vw, 1.7rem);
    font-weight: 400;
    margin-bottom: 0.6rem;
}
.roadmap-grid .copy { font-size: 0.92rem; line-height: 1.6; color: var(--dim); }

.about-disclaimer {
    border-left: 2px solid var(--warm);
    padding: 1.1rem 1.4rem;
    color: var(--muted);
    background: #10130f;
    line-height: 1.65;
    font-size: 0.95rem;
}
.about-disclaimer strong { color: var(--text); }

/* ---- Responsive ---------------------------------------------------------- */

@media (max-width: 1150px) {
    .wildid-header { gap: 1rem; }
    .wildid-brand-sub, .wildid-baseline { display: none; }
}

@media (max-width: 900px) {
    .wildid-header {
        flex-wrap: wrap;
        row-gap: 0;
        padding-top: 0.8rem;
        padding-bottom: 0;
    }
    .wildid-brand { flex: 1 1 auto; }
    .wildid-status { flex: 0 0 auto; }
    .wildid-nav {
        order: 3;
        flex: 1 1 100%;
        justify-content: flex-start;
        gap: 1.25rem;
        margin-top: 0.4rem;
        overflow-x: auto;
        scrollbar-width: none;
    }
    .wildid-nav::-webkit-scrollbar { display: none; }

    .step-grid,
    .evidence-grid,
    .status-grid,
    .about-intro,
    .about-grid,
    .about-method,
    .about-abstain,
    .roadmap-grid { grid-template-columns: minmax(0, 1fr); }

    .step { border-right: none; border-bottom: 1px solid var(--line); }
    .step:last-child { border-bottom: none; }
    .about-card-wide { grid-row: auto; }
    .about-intro-copy { padding-bottom: 0; }
    .about-intro { gap: 1.5rem; }

    /* Stack the outer Identify columns before Streamlit's own 640px breakpoint. */
    .st-key-wid-identify [data-testid="stHorizontalBlock"]:not([data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"]) {
        flex-wrap: wrap !important;
    }
    .st-key-wid-identify [data-testid="stHorizontalBlock"]:not([data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"]) > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }
}

@media (max-width: 560px) {
    .metric-strip { grid-template-columns: minmax(0, 1fr); }
    .metric {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
        border-right: none;
        border-bottom: 1px solid var(--line);
    }
    .metric:last-child { border-bottom: none; }
    .metric-label { margin-bottom: 0; }
    .method-list > div { grid-template-columns: 2.5rem minmax(0, 1fr); }
    .method-list small { grid-column: 2; margin-top: -0.6rem; }
    .schematic-grid { grid-template-columns: minmax(0, 1fr); }
    .wildid-nav { gap: 1rem; }
    .hero-actions a { flex: 1 1 100%; }
    .hero-content { padding-top: 3.25rem; padding-bottom: 3.5rem; }
}
"""


def inject_css():
    st.markdown("<style>" + APP_CSS + "</style>", unsafe_allow_html=True)


def render_html(markup):
    """Render static HTML safely through st.markdown.

    Lines are stripped and blank lines removed so Markdown never treats
    indented or separated HTML as a code block.
    """
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def page_container(name):
    """Keyed container; gives each page a stable, shared width/padding wrapper."""
    return st.container(key=f"wid-{name}")


def page_head(eyebrow, title, lede):
    render_html(
        f"""
        <div class="page-head">
        <div class="eyebrow">{eyebrow}</div>
        <div class="page-title" role="heading" aria-level="1">{title}</div>
        <div class="lede">{lede}</div>
        </div>
        """
    )


def render_brandbar(active=None):
    def link(label):
        current = ' aria-current="page"' if label == active else ""
        return f'<a href="?page={label}" target="_self"{current}>{label}</a>'

    nav_links = "".join(
        link(label) for label in ["Overview", "Identify", "Individuals", "Sightings", "About"]
    )
    render_html(
        f"""
        <header class="wildid-header">
        <a class="wildid-brand" href="?page=Overview" target="_self">
        <span class="wildid-brand-name">WILDID</span>
        <span class="wildid-brand-sub">Individual Wildlife Re-Identification</span>
        </a>
        <nav class="wildid-nav" aria-label="Primary navigation">{nav_links}</nav>
        <div class="wildid-status">
        <span class="wildid-prototype">PROTOTYPE</span>
        <span class="wildid-baseline">R001 BASELINE</span>
        </div>
        </header>
        """
    )


def fmt_score(value):
    return f"{value:.4f}"


def render_overview():
    render_html(
        """
        <div class="hero-wrap">
        <div class="hero-image" aria-hidden="true"></div>
        <div class="hero-overlay" aria-hidden="true"></div>
        <div class="hero-content">
        <div class="eyebrow hero-eyebrow">Conservation intelligence · prototype</div>
        <div class="hero-title" role="heading" aria-level="1">Every individual,<br><em>leaves a visual signature.</em></div>
        <div class="lede hero-lede">WildID retrieves the most similar known individuals for a camera-trap image and declines to name one when the evidence is not strong enough.</div>
        <div class="hero-actions">
        <a class="hero-action-primary" href="?page=Identify" target="_self">Open Identify</a>
        <a class="hero-action-secondary" href="#how-it-works">How it works</a>
        </div>
        </div>
        </div>
        """
    )

    with page_container("overview"):
        render_html(
            f"""
            <div class="step-grid" id="how-it-works">
            <div class="step">
            <div class="step-no">01</div>
            <div class="step-title">Query</div>
            <div class="step-copy">A camera-trap frame is embedded by the re-identification model.</div>
            </div>
            <div class="step">
            <div class="step-no">02</div>
            <div class="step-title">Retrieve</div>
            <div class="step-copy">Gallery individuals are ranked by cosine similarity.</div>
            </div>
            <div class="step">
            <div class="step-no">03</div>
            <div class="step-title">Decide</div>
            <div class="step-copy">At or above {THRESHOLD_LABEL} the top candidate is a Match; below it, No Reliable Match.</div>
            </div>
            </div>
            """
        )

        render_html(
            f"""
            <div class="evidence-grid mt-section">
            <div class="panel">
            <div class="panel-title">System evidence</div>
            <div class="kv-plain">
            <div class="kv-row"><span>Model</span><strong>MegaDescriptor-T-224</strong></div>
            <div class="kv-row"><span>Retrieval</span><strong>Cosine similarity</strong></div>
            <div class="kv-row"><span>Decision threshold</span><strong>{THRESHOLD_LABEL}</strong></div>
            <div class="kv-row"><span>Decision states</span><strong>Match / No Reliable Match</strong></div>
            </div>
            </div>
            <div class="panel">
            <div class="panel-title">Prototype evidence · R001 ATRW baseline</div>
            <div class="metric-strip">
            <div class="metric"><div class="metric-label">TOP-1</div><div class="metric-value">97.75%</div></div>
            <div class="metric"><div class="metric-label">TOP-5</div><div class="metric-value">99.44%</div></div>
            <div class="metric"><div class="metric-label">MAP</div><div class="metric-value">94.22%</div></div>
            </div>
            <div class="kv-plain">
            <div class="kv-row"><span>Identities</span><strong>107</strong></div>
            <div class="kv-row"><span>Gallery images</span><strong>642</strong></div>
            <div class="kv-row"><span>Query images</span><strong>1,245</strong></div>
            <div class="kv-row"><span>Demo decision threshold</span><strong>{THRESHOLD_LABEL}</strong></div>
            </div>
            <div class="fineprint">These are offline evaluation results on the evaluated ATRW split. They are not proof of production accuracy or field deployment readiness, and {THRESHOLD_LABEL} is a prototype demo decision rule, not a universal production confidence threshold.</div>
            </div>
            </div>
            """
        )


def render_identify(model, gallery_embeddings, gallery_ids, gallery_filenames):
    with page_container("identify"):
        page_head(
            "01 — query",
            "Identify an individual",
            "Upload a camera-trap image to retrieve the closest known individual candidates.",
        )

        left, right = st.columns([1.05, 0.95], gap="large")

        image = None
        image_name = None
        image_source = None

        with left:
            uploaded = st.file_uploader(
                "Drop a camera-trap image",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible",
            )

            render_html('<div class="section-label">Demo images</div>')
            demo_a, demo_b = st.columns(2)

            if "selected_demo" not in st.session_state:
                st.session_state.selected_demo = None

            with demo_a:
                if st.button("002253.jpg · MATCH", use_container_width=True):
                    st.session_state.selected_demo = "002253.jpg"

            with demo_b:
                if st.button("001454.jpg · REJECT", use_container_width=True):
                    st.session_state.selected_demo = "001454.jpg"

            if uploaded is not None:
                image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
                image_name = uploaded.name
                image_source = "Uploaded file"
            elif st.session_state.selected_demo:
                image_name = st.session_state.selected_demo
                image = get_demo_image(image_name)
                image_source = "Demo image"

            if image is not None:
                if uploaded is not None and st.session_state.selected_demo:
                    render_html(
                        '<div class="notice">An uploaded file takes priority over the selected demo image.</div>'
                    )
                st.image(image, use_container_width=True)
                safe_name = escape(str(image_name))
                render_html(
                    f"""
                    <div class="img-caption">
                    <span>{escape(image_source)}</span>
                    <b title="{safe_name}">{safe_name}</b>
                    </div>
                    """
                )
                if st.button("Identify Individual", key="identify_button", type="primary"):
                    with st.spinner("Comparing against the gallery..."):
                        result = search_gallery(
                            image,
                            model,
                            gallery_embeddings,
                            gallery_ids,
                            gallery_filenames,
                        )
                    st.session_state.result = result
                    st.session_state.result_image = image
                    st.session_state.result_name = image_name
                    st.rerun()

        with right:
            result = st.session_state.get("result")

            if result is None:
                render_html(
                    f"""
                    <div class="panel">
                    <div class="panel-title">02 — result</div>
                    <div class="panel-heading">Awaiting a query image.</div>
                    <div class="fineprint">Upload an image or select a demo image, then press Identify Individual. The result will contain:</div>
                    <div class="list-rows">
                    <div class="list-row"><span><em>01</em>Best candidate</span><small>Closest known individual</small></div>
                    <div class="list-row"><span><em>02</em>Similarity score</span><small>Cosine similarity, 0–1</small></div>
                    <div class="list-row"><span><em>03</em>Top candidates</span><small>Five ranked gallery identities</small></div>
                    <div class="list-row"><span><em>04</em>Decision</span><small>Match / No Reliable Match at {THRESHOLD_LABEL}</small></div>
                    </div>
                    </div>
                    """
                )
            else:
                decision = result["decision"]
                result_name = str(st.session_state.get("result_name") or "query image")
                is_stale = image_name is not None and image_name != st.session_state.get("result_name")
                top_identity = result["top1_identity"]
                top_score = fmt_score(result["top1_score"])

                stale_html = ""
                if is_stale:
                    stale_html = (
                        '<div class="notice" style="margin-top:0.75rem;">'
                        "The result below belongs to the previous query. "
                        "Press Identify Individual to compare the current image."
                        "</div>"
                    )

                if decision == "MATCH":
                    card_class = "is-match"
                    state_label = "MATCH"
                    headline = f"Individual #{top_identity}"
                    note = "Strong visual evidence supports this individual match."
                else:
                    card_class = "is-no"
                    state_label = "NO RELIABLE MATCH"
                    headline = "Evidence below threshold"
                    note = (
                        "The available visual evidence is insufficient for a confident individual identification. "
                        f"Closest retrieved candidate: #{top_identity} (not accepted as a match)."
                    )

                render_html(
                    f"""
                    <div class="result-meta">02 — result · <b title="{escape(result_name)}">{escape(result_name)}</b></div>
                    {stale_html}
                    <div class="result-card {card_class}" style="margin-top:0.75rem;">
                    <div class="result-state">{state_label}</div>
                    <div class="result-identity">{headline}</div>
                    <div class="result-score"><b>{top_score}</b>cosine similarity · threshold {THRESHOLD_LABEL}</div>
                    <div class="fineprint">{note}</div>
                    </div>
                    """
                )

                bar_top = "var(--green)" if decision == "MATCH" else "var(--amber)"
                rows = []
                for candidate in result["candidates"]:
                    pct = max(0.0, min(100.0, candidate["similarity"] * 100))
                    top_cls = " is-top" if candidate["rank"] == 1 else ""
                    rank_txt = f"{candidate['rank']:02d}"
                    identity_txt = f"#{candidate['identity']}"
                    file_txt = escape(str(candidate["filename"]))
                    score_txt = fmt_score(candidate["similarity"])
                    rows.append(
                        f"""
                        <div class="cand-row{top_cls}">
                        <div class="cand-rank">{rank_txt}</div>
                        <div class="cand-id"><span class="cand-id-main">{identity_txt}</span><span class="cand-file" title="{file_txt}">{file_txt}</span></div>
                        <div class="cand-bar"><div class="cand-bar-fill" style="width:{pct:.2f}%;"></div><div class="cand-bar-tick"></div></div>
                        <div class="cand-score">{score_txt}</div>
                        </div>
                        """
                    )

                render_html(
                    f"""
                    <div class="cand-panel" style="--bar-top:{bar_top};">
                    <div class="cand-title"><div class="panel-title">Top candidates</div></div>
                    {"".join(rows)}
                    <div class="cand-legend">Cosine similarity, bar spans 0–1. Marker shows the {THRESHOLD_LABEL} prototype threshold. Scores are not calibrated probabilities.</div>
                    </div>
                    """
                )

                # Optional: show the closest gallery reference image when it is available locally.
                try:
                    ref_file = Path(str(result["candidates"][0]["filename"])).name
                    if (IMAGE_DIR / ref_file).exists():
                        render_html('<div class="section-label">Closest gallery reference</div>')
                        st.image(get_demo_image(ref_file), use_container_width=True)
                        render_html(
                            f'<div class="img-caption"><b title="{escape(ref_file)}">{escape(ref_file)}</b></div>'
                        )
                except Exception:
                    pass

        render_html(
            f"""
            <div class="fineprint" style="margin-top:2rem;">Prototype threshold: {THRESHOLD_LABEL}. This is an R001 evaluation-derived demo rule, not a universal production confidence threshold. Similarity scores are cosine similarities between embeddings, not calibrated probabilities.</div>
            """
        )


def render_individuals():
    with page_container("individuals"):
        page_head(
            "02 — individual intelligence",
            "Individual profiles",
            "A future layer for persistent individual records, evidence, and observation context.",
        )
        render_html(
            """
            <div class="status-grid">
            <div class="panel">
            <div class="panel-title">Implementation status</div>
            <div class="h-card" role="heading" aria-level="2">No persistent individual records yet.</div>
            <div class="copy">This prototype does not store individual records. The Identify page compares a query image with the reference gallery loaded by the app and returns ranked candidate identities for that single query. No profile is created, edited, or saved, and nothing on this page is generated from data.</div>
            <div class="mt-block">
            <div class="status-row"><span>Gallery retrieval on the Identify page<small>Ranked candidates and a Match / No Reliable Match decision</small></span><span class="tag is-ok">WORKING</span></div>
            <div class="status-row"><span>Persistent individual registry</span><span class="tag">NOT IMPLEMENTED</span></div>
            <div class="status-row"><span>Individual profile pages and evidence history</span><span class="tag">NOT IMPLEMENTED</span></div>
            </div>
            </div>
            <div class="panel">
            <div class="panel-title">Planned profile layout · illustrative</div>
            <div class="schematic-grid">
            <div class="schematic-media" role="img" aria-label="Placeholder for reference images">Reference images</div>
            <div>
            <div class="schematic-field"><span>Individual ID</span><i>—</i></div>
            <div class="schematic-field"><span>Retrieval evidence</span><i>—</i></div>
            <div class="schematic-field"><span>Linked observations</span><i>—</i></div>
            </div>
            </div>
            <div class="fineprint">A planned design only. It contains no data and no example animals; profiles will appear here once a persistent registry exists and has been validated.</div>
            </div>
            </div>
            """
        )


def render_sightings():
    with page_container("sightings"):
        page_head(
            "03 — observation log",
            "Sighting history",
            "Confirmed matches can become dated observations in the full WildID system. No sighting records are fabricated in this prototype.",
        )
        render_html(
            """
            <div class="status-grid">
            <div class="panel compare">
            <div class="panel-title">Retrieval result vs. field observation</div>
            <div class="compare-row is-head"><div></div><div>Identify result · available</div><div>Sighting record · not implemented</div></div>
            <div class="compare-row"><div class="compare-key">Source</div><div>A single uploaded or demo image</div><div>A dated camera-trap event</div></div>
            <div class="compare-row"><div class="compare-key">Contains</div><div>Cosine similarity scores, ranked candidates, a decision</div><div>Timestamp, location, and reviewed confirmation</div></div>
            <div class="compare-row"><div class="compare-key">Stored</div><div>Current session only; not written to any log</div><div>Persistent observation history</div></div>
            <div class="fineprint">A retrieval result is a similarity-based suggestion for one image. It is not a confirmed, dated field observation.</div>
            </div>
            <div class="panel">
            <div class="panel-title">Not yet implemented</div>
            <div class="status-row"><span>Persistent sighting log</span><span class="tag">NOT IMPLEMENTED</span></div>
            <div class="status-row"><span>Spatial context and movement history</span><span class="tag">NOT IMPLEMENTED</span></div>
            <div class="status-row"><span>Live camera-trap integration</span><span class="tag">NOT IMPLEMENTED</span></div>
            <div class="status-row"><span>Conservation alerts</span><span class="tag">NOT IMPLEMENTED</span></div>
            <div class="fineprint">No timestamps, coordinates, routes, or maps are shown because none are stored by this prototype.</div>
            </div>
            </div>
            """
        )


def render_about():
    with page_container("about"):
        page_head(
            "About WildID · Project brief",
            "Individual wildlife recognition, built around evidence.",
            "WildID addresses the gap between <strong>detecting an animal</strong> and determining whether the animal "
            "has been seen before. The prototype compares a camera-trap image with known individuals, ranks visual "
            "candidates, and can return <strong>No Reliable Match</strong> instead of forcing an identity.",
        )

        render_html(
            """
            <section class="about-intro">
            <div class="about-intro-copy">
            <div class="panel-title">01 · Why this matters</div>
            <div class="h-section" role="heading" aria-level="2">From “a tiger was detected” to “which tiger is this?”</div>
            <div class="copy">Wildlife monitoring produces large volumes of camera-trap images, but identifying the same individual across observations is a different problem from species detection. WildID is designed around that individual-level question. It turns a query image into a visual embedding, compares it against a gallery of known individuals, ranks the closest candidates, and exposes the evidence used for the decision.</div>
            </div>
            <div class="about-intro-image"><div class="about-image-label">Tiger · wildlife reference image</div></div>
            </section>
            """
        )

        render_html(
            """
            <section class="about-grid mt-section">
            <article class="about-card about-card-wide">
            <div class="panel-title">02 · Core distinction</div>
            <div class="h-card" role="heading" aria-level="2">Detection is not identity.</div>
            <div class="copy">A detector can establish that an animal appears in a frame. Re-identification asks a harder question: does this animal correspond to one of the known individuals already represented in the gallery? WildID keeps those concepts separate. The system does not treat the nearest visual match as automatically correct; it ranks candidates and uses a confidence rule to decide whether the evidence is strong enough for the prototype decision.</div>
            <div class="copy">That evidence-first design is important for conservation workflows because an incorrect individual assignment can contaminate later records. A valid system therefore needs an explicit way to say <strong>No Reliable Match</strong> when the available evidence is insufficient.</div>
            </article>
            <article class="about-card">
            <div class="panel-title">03 · What works today</div>
            <div class="h-card" role="heading" aria-level="2">Tiger-first and deliberately small.</div>
            <div class="copy">The current implementation is a <strong>small working tiger re-identification prototype</strong>. It demonstrates the actual retrieval pipeline rather than claiming a finished conservation platform.</div>
            </article>
            <article class="about-card">
            <div class="panel-title">04 · What it is not</div>
            <div class="h-card" role="heading" aria-level="2">Not a finished deployment.</div>
            <div class="copy">Individual profiles, longitudinal sighting history, spatial intelligence, live camera integration, and broad multi-species validation are not being presented as completed features. They are the next system layers to build and validate.</div>
            </article>
            </section>
            """
        )

        render_html(
            f"""
            <section class="about-method mt-section">
            <div>
            <div class="panel-title">05 · Working methodology</div>
            <div class="h-section" role="heading" aria-level="2">Query → Retrieve → Decide</div>
            <div class="copy">A query image is converted into an embedding using MegaDescriptor-T-224. That embedding is compared with gallery embeddings using cosine similarity. The prototype ranks the strongest candidates and applies the evaluation-derived {THRESHOLD_LABEL} rule used for the current demo decision.</div>
            <div class="copy">This is a retrieval prototype, not a claim that {THRESHOLD_LABEL} is a universal production confidence threshold. Thresholds would need species-specific validation and field testing before deployment.</div>
            </div>
            <div class="method-list">
            <div><span>01</span><strong>Query</strong><small>Camera-trap image</small></div>
            <div><span>02</span><strong>Embed</strong><small>Visual representation</small></div>
            <div><span>03</span><strong>Retrieve</strong><small>Rank known individuals</small></div>
            <div><span>04</span><strong>Decide</strong><small>Match or No Reliable Match</small></div>
            </div>
            </section>
            """
        )

        render_html(
            f"""
            <section class="mt-section">
            <div class="panel-title">06 · R001 ATRW baseline · measured prototype evidence</div>
            <div class="metric-strip">
            <div class="metric"><div class="metric-label">TOP-1</div><div class="metric-value">97.75%</div></div>
            <div class="metric"><div class="metric-label">TOP-5</div><div class="metric-value">99.44%</div></div>
            <div class="metric"><div class="metric-label">MAP</div><div class="metric-value">94.22%</div></div>
            </div>
            <div class="about-stats"><span><b>107</b> identities</span><span><b>642</b> gallery images</span><span><b>1,245</b> query images</span><span><b>{THRESHOLD_LABEL}</b> demo threshold</span></div>
            <div class="fineprint">These are measured R001 offline evaluation results on the ATRW split used for this prototype. They are evidence that the current retrieval prototype works on the evaluated split, not live production accuracy and not proof of deployment readiness.</div>
            </section>
            """
        )

        render_html(
            """
            <section class="about-abstain mt-section">
            <div>
            <div class="panel-title">07 · Evidence over forced answers</div>
            <div class="h-section" role="heading" aria-level="2">“No Reliable Match” is a feature, not a failure.</div>
            <div class="copy">If the strongest candidate does not meet the prototype rule, WildID does not invent certainty. This creates a safer decision boundary for a future conservation system and makes uncertainty visible to the user instead of hiding it behind a forced identity.</div>
            </div>
            <div class="abstain-badge"><span>DECISION STATE</span><strong>NO RELIABLE MATCH</strong><small>Evidence below threshold</small></div>
            </section>
            """
        )

        render_html(
            """
            <section class="mt-section">
            <div class="panel-title">08 · Expansion beyond the current prototype</div>
            <div class="roadmap-grid">
            <div><span>NOW</span><strong>Working tiger Re-ID</strong><div class="copy">Real image retrieval, ranked candidates, measured R001 evaluation, and an explicit abstention state.</div></div>
            <div><span>NEXT</span><strong>Other relevant animals</strong><div class="copy">The architecture is intended to extend to other relevant wildlife species, with species-appropriate datasets and validation.</div></div>
            <div><span>FUTURE</span><strong>Conservation intelligence</strong><div class="copy">Validated individual profiles, sighting history, spatial context, and integration with wildlife monitoring workflows.</div></div>
            </div>
            </section>
            """
        )

        render_html(
            """
            <div class="about-disclaimer mt-section"><strong>Prototype scope:</strong> WildID is currently a small, working tiger re-identification prototype. The broader multi-species conservation intelligence system is the intended direction. Other relevant animals can be supported as suitable datasets, models, and validation are added; those future layers are not being represented here as already deployed.</div>
            """
        )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

inject_css()

valid_pages = {"Overview", "Identify", "Individuals", "Sightings", "About"}
query_page = st.query_params.get("page")
if query_page in valid_pages:
    st.session_state.page = query_page
elif "page" not in st.session_state:
    st.session_state.page = "Overview"

page = st.session_state.page

render_brandbar(page)

if page == "Overview":
    render_overview()

elif page == "Identify":
    try:
        model = load_model()
        gallery_embeddings, gallery_ids, gallery_filenames = load_gallery()
        render_identify(model, gallery_embeddings, gallery_ids, gallery_filenames)
    except Exception as exc:
        st.error("WildID could not initialize the local inference assets.")
        st.code(str(exc))

elif page == "Individuals":
    render_individuals()

elif page == "Sightings":
    render_sightings()

elif page == "About":
    render_about()
