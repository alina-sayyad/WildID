
import io
import base64
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


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&family=Playfair+Display:ital,wght@0,400;0,500;1,400;1,500&display=swap');

        :root {
            --bg: #0b0d0b;
            --surface: #111410;
            --surface-2: #151915;
            --line: #2b302b;
            --text: #eee9dc;
            --muted: #a6a99f;
            --warm: #d9c9a9;
            --green: #829276;
            --green-bright: #a2b494;
            --amber: #c9ad78;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(11, 13, 11, 0.96);
        }

        .block-container {
            max-width: none !important;
            padding: 0 0 3rem 0 !important;
        }

        .wildid-header {
            position: relative;
            z-index: 20;
            width: 100%;
            box-sizing: border-box;
            padding: 0.85rem 3.6rem;
            margin: 0;
            border-bottom: 1px solid rgba(232,225,207,0.10);
            background: linear-gradient(180deg, rgba(8,10,8,0.98), rgba(8,10,8,0.74), rgba(8,10,8,0.08));
            display: grid;
            grid-template-columns: minmax(320px, 1.25fr) minmax(430px, 1fr) auto;
            align-items: center;
            column-gap: 1.4rem;
        }

        .wildid-header-nav-space { min-width: 0; }

        .wildid-nav {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: clamp(1.2rem, 2.4vw, 3rem);
            min-width: 0;
        }

        .wildid-nav a {
            color: #f0ede5 !important;
            text-decoration: none !important;
            font-family: "DM Sans", sans-serif;
            font-size: 0.78rem;
            font-weight: 500;
            line-height: 1;
            white-space: nowrap;
            padding: 0.65rem 0.15rem;
            text-shadow: 0 1px 8px rgba(0,0,0,0.8);
            transition: color 120ms ease, border-color 120ms ease;
            border-bottom: 1px solid transparent;
        }

        .wildid-nav a:hover {
            color: #ffffff !important;
            border-bottom-color: #d9c9a9;
        }

        .wildid-status {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 1rem;
            min-width: 0;
        }


        .wildid-brand {
            display: flex;
            text-decoration: none !important;
            align-items: baseline;
            gap: 1.1rem;
            min-width: 0;
        }

        .wildid-brand-name {
            font-family: "DM Mono", monospace;
            font-size: 0.92rem;
            font-weight: 500;
            letter-spacing: 0.34em;
            color: #f2eee5 !important;
            text-shadow: 0 1px 10px rgba(0,0,0,0.65);
            white-space: nowrap;
        }

        .wildid-brand-sub {
            color: #d1d0c8 !important;
            text-shadow: 0 1px 8px rgba(0,0,0,0.65);
            font-family: "DM Sans", sans-serif;
            font-size: 0.72rem;
            white-space: nowrap;
        }

        .wildid-nav .stButton > button {
            background: transparent !important;
            border: 0 !important;
            color: #b7bab2 !important;
            font-family: "DM Sans", sans-serif;
            font-size: 0.76rem;
            font-weight: 400;
            padding: 0.25rem 0 !important;
            min-height: 1.7rem;
            white-space: nowrap;
            box-shadow: none !important;
        }

        .wildid-nav .stButton > button:hover {
            background: transparent !important;
            border: 0 !important;
            color: #f0ece1 !important;
        }

        .wildid-prototype {
            justify-self: end;
            border: 1px solid #363a34;
            padding: 0.52rem 0.72rem;
            color: #b8b2a3;
            font-family: "DM Mono", monospace;
            font-size: 0.61rem;
            letter-spacing: 0.14em;
            white-space: nowrap;
        }

        .wildid-baseline {
            justify-self: end;
            color: #d5c8ae;
            font-family: "DM Mono", monospace;
            font-size: 0.61rem;
            letter-spacing: 0.14em;
            white-space: nowrap;
        }

        .eyebrow {
            font-family: "DM Mono", monospace;
            font-size: 0.68rem;
            letter-spacing: 0.18em;
            color: #8b9187;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }

        h1, h2, h3 {
            font-family: "Playfair Display", Georgia, serif !important;
            font-weight: 400 !important;
            color: var(--text) !important;
        }

        .hero-title {
            font-family: "Playfair Display", Georgia, serif;
            font-size: clamp(3rem, 5.5vw, 6rem);
            line-height: 0.96;
            letter-spacing: -0.045em;
            max-width: 950px;
            color: var(--text);
            margin: 0;
        }

        .hero-title em {
            color: var(--warm);
        }

        .lede {
            max-width: 720px;
            color: #a9ada4;
            font-family: "DM Sans", sans-serif;
            font-size: 1.03rem;
            line-height: 1.65;
            margin-top: 1.5rem;
        }

        .section-label {
            font-family: "DM Mono", monospace;
            color: #858b81;
            font-size: 0.68rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .rule {
            height: 1px;
            background: var(--line);
            margin: 2rem 0;
        }

        .step-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            border-top: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
            margin: 4rem 3.6rem 0;
        }

        .step {
            min-height: 190px;
            padding: 1.5rem 2rem 2rem 2rem;
            border-right: 1px solid var(--line);
        }

        .step:last-child {
            border-right: none;
        }

        .step-no {
            font-family: "DM Mono", monospace;
            font-size: 0.68rem;
            color: #747a70;
            margin-bottom: 2rem;
        }

        .step-title {
            font-family: "Playfair Display", Georgia, serif;
            font-size: 1.7rem;
            color: var(--text);
            margin-bottom: 0.7rem;
        }

        .step-copy {
            color: #92978e;
            line-height: 1.55;
        }

        .panel {
            border: 1px solid var(--line);
            background: var(--surface);
            padding: 1.5rem;
        }

        .panel-title {
            font-family: "DM Mono", monospace;
            font-size: 0.68rem;
            letter-spacing: 0.16em;
            color: #858b81;
            text-transform: uppercase;
            margin-bottom: 1.2rem;
        }

        .result-match {
            border-left: 3px solid var(--green);
            background: #111811;
            padding: 1.3rem 1.4rem;
        }

        .result-no {
            border-left: 3px solid var(--amber);
            background: #17140e;
            padding: 1.3rem 1.4rem;
        }

        .decision {
            font-family: "DM Mono", monospace;
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            margin-bottom: 0.65rem;
        }

        .identity {
            font-family: "Playfair Display", Georgia, serif;
            font-size: 2.5rem;
            color: var(--text);
        }

        .score {
            font-family: "DM Mono", monospace;
            font-size: 1.1rem;
            color: var(--warm);
        }

        .candidate-row {
            display: grid;
            grid-template-columns: 46px 110px 1fr 90px;
            align-items: center;
            gap: 1rem;
            border-top: 1px solid var(--line);
            padding: 0.85rem 0;
        }

        .candidate-rank,
        .candidate-id,
        .candidate-score {
            font-family: "DM Mono", monospace;
            font-size: 0.72rem;
        }

        .candidate-rank {
            color: #72786e;
        }

        .candidate-id {
            color: var(--text);
        }

        .candidate-score {
            color: var(--warm);
            text-align: right;
        }

        .candidate-bar {
            height: 2px;
            background: #2a2f2a;
        }

        .candidate-bar-fill {
            height: 2px;
            background: var(--green);
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            border: 1px solid var(--line);
            background: var(--surface);
        }

        .metric {
            padding: 1.3rem 1.4rem;
            border-right: 1px solid var(--line);
        }

        .metric:last-child {
            border-right: none;
        }

        .metric-label {
            font-family: "DM Mono", monospace;
            font-size: 0.62rem;
            letter-spacing: 0.14em;
            color: #747a70;
            margin-bottom: 0.55rem;
        }

        .metric-value {
            font-family: "Playfair Display", Georgia, serif;
            font-size: 2rem;
            color: var(--text);
        }

        .fineprint {
            color: #6f756c;
            font-size: 0.75rem;
            line-height: 1.5;
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed #353a34;
            background: #0f120f;
        }

        div[data-testid="stFileUploader"] section {
            background: transparent;
        }

        .stButton > button {
            border: 1px solid #3a3f38;
            border-radius: 0;
            background: #d9c9a9;
            color: #0b0d0b;
            font-family: "DM Sans", sans-serif;
            font-weight: 600;
            padding: 0.65rem 1.25rem;
        }

        .stButton > button:hover {
            border-color: #d9c9a9;
            background: #eee2c8;
            color: #0b0d0b;
        }

        img {
            border-radius: 0 !important;
        }

        @media (max-width: 900px) {
            .block-container {
                padding: 0 0 2rem 0 !important;
            }

            .hero {
                min-height: 500px;
            }

            .hero-content {
                padding: 3rem 2rem 2.5rem 2rem;
            }

            .hero-title-large {
                font-size: clamp(3rem, 12vw, 5rem);
            }

            .hero-actions {
                flex-wrap: wrap;
            }

            .step-grid {
                grid-template-columns: 1fr;
            }

            .step {
                border-right: none;
                border-bottom: 1px solid var(--line);
            }

            .metric-strip {
                grid-template-columns: 1fr;
            }

            .metric {
                border-right: none;
                border-bottom: 1px solid var(--line);
            }
        }

        [data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
        [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
        #MainMenu, footer { display: none !important; }

        .wildid-header {
            min-height: 78px !important;
            padding: 0 5.4vw !important;
            grid-template-columns: minmax(360px, 1fr) auto minmax(280px, 1fr) !important;
            background: #090b09 !important;
            border-bottom: 1px solid rgba(238,232,218,.14) !important;
        }
        .wildid-brand-name, .wildid-brand-sub, .wildid-nav a, .wildid-prototype, .wildid-baseline {
            opacity: 1 !important;
            text-shadow: 0 1px 9px rgba(0,0,0,.7) !important;
        }
        .wildid-brand-name { color:#f4f0e6 !important; -webkit-text-fill-color:#f4f0e6 !important; }
        .wildid-brand-sub { color:#d1d0c8 !important; -webkit-text-fill-color:#d1d0c8 !important; }
        .wildid-nav a { color:#eee9dc !important; -webkit-text-fill-color:#eee9dc !important; }
        .wildid-nav a:hover, .wildid-nav a.active { color:#fffdf6 !important; -webkit-text-fill-color:#fffdf6 !important; }
        .wildid-prototype { color:#d9cdb4 !important; -webkit-text-fill-color:#d9cdb4 !important; }
        .wildid-baseline { color:#cfc5b1 !important; -webkit-text-fill-color:#cfc5b1 !important; }

        .hero-wrap { width:100% !important; min-height:calc(100vh - 78px) !important; height:820px !important; border:0 !important; border-radius:0 !important; }
        .hero-image { inset:0 !important; background-position:center center !important; filter:saturate(.44) contrast(.95) brightness(.60) !important; }
        .hero-overlay { inset:0 !important; background:linear-gradient(90deg,rgba(7,9,7,.88) 0%,rgba(7,9,7,.63) 31%,rgba(7,9,7,.18) 70%,rgba(7,9,7,.32) 100%),linear-gradient(180deg,rgba(7,9,7,.38) 0%,rgba(7,9,7,.02) 34%,rgba(7,9,7,.18) 64%,rgba(7,9,7,.76) 91%,#090b09 100%) !important; }

        .identify-shell { width:min(1240px, calc(100% - 8rem)) !important; max-width:1240px !important; margin:0 auto !important; padding:5.2rem 0 7rem !important; box-sizing:border-box !important; }
        .identify-heading { margin-bottom:3.2rem !important; }
        .identify-grid { display:grid !important; grid-template-columns:minmax(0,1.03fr) minmax(0,.97fr) !important; gap:4.5rem !important; align-items:start !important; }
        .identify-upload-wrap, .identify-result-wrap { min-width:0 !important; }
        div[data-testid="stFileUploader"] { border:1px dashed #454940 !important; background:#0e110e !important; border-radius:0 !important; padding:1.15rem !important; }
        div[data-testid="stFileUploader"] label { color:#dedbd2 !important; }
        div[data-testid="stFileUploader"] button { background:#ded0b2 !important; color:#0a0c0a !important; border:0 !important; border-radius:0 !important; }

        .page-shell { width:min(1180px,calc(100% - 11vw)) !important; margin:0 auto !important; padding:5.2rem 0 7rem !important; }

        @media (max-width:1050px) {
            .wildid-header { grid-template-columns:1fr !important; padding:.9rem 5vw 0 !important; }
            .wildid-brand { justify-content:center !important; }
            .wildid-nav { height:54px !important; overflow-x:auto !important; justify-content:flex-start !important; }
            .wildid-nav a { height:54px !important; }
            .wildid-status { display:none !important; }
            .identify-grid { grid-template-columns:1fr !important; gap:2.5rem !important; }
        }
        @media (max-width:700px) {
            .identify-shell, .page-shell { width:calc(100% - 10vw) !important; max-width:none !important; }
            .hero-wrap { height:680px !important; min-height:680px !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brandbar():
    st.markdown(
        """
        <header class="wildid-header">
            <a class="wildid-brand" href="?page=Overview" target="_self">
                <span class="wildid-brand-name">WILDID</span>
                <span class="wildid-brand-sub">Individual Wildlife Re-Identification</span>
            </a>
            <nav class="wildid-nav" aria-label="Primary navigation">
                <a href="?page=Overview" target="_self">Overview</a>
                <a href="?page=Identify" target="_self">Identify</a>
                <a href="?page=Individuals" target="_self">Individuals</a>
                <a href="?page=Sightings" target="_self">Sightings</a>
                <a href="?page=About" target="_self">About</a>
            </nav>
            <div class="wildid-status">
                <span class="wildid-prototype">PROTOTYPE</span>
                <span class="wildid-baseline">R001 BASELINE</span>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_overview():
    st.markdown(
        '''
        <div class="hero-wrap">
            <div class="hero-image"></div>
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="eyebrow hero-eyebrow">Conservation intelligence · prototype</div>
                <div class="hero-title">
                    Every individual,<br>
                    <em>leaves a visual signature.</em>
                </div>
                <div class="lede hero-lede">
                    WildID retrieves the most similar known individuals for a camera-trap image
                    and declines to name one when the evidence is not strong enough.
                </div>
                <div class="hero-actions">
                    <a class="hero-action-primary" href="?page=Identify" target="_self">Open Identify</a>
                    <a class="hero-action-secondary" href="#how-it-works">How it works</a>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''
        <style>
        .hero-wrap {
            position: relative;
            min-height: 760px;
            width: 100%;
            overflow: hidden;
            background: #0b0d0b;
            margin: 0;
        }

        .hero-image {
            position: absolute;
            inset: -1.5%;
            background-image: url("https://images.unsplash.com/photo-1698382439843-ca033a6079c0?fm=jpg&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&ixlib=rb-4.1.0&q=85&w=2600");
            background-size: cover;
            background-position: center 42%;
            filter: saturate(0.48) contrast(0.94) brightness(0.56);
            transform: scale(1.02);
        }

        .hero-overlay {
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(7,9,7,0.90) 0%, rgba(7,9,7,0.62) 32%, rgba(7,9,7,0.16) 72%, rgba(7,9,7,0.30) 100%),
                linear-gradient(180deg, rgba(7,9,7,0.30) 0%, rgba(7,9,7,0.02) 38%, rgba(7,9,7,0.34) 70%, #0b0d0b 100%);
        }

        .hero-content {
            position: relative;
            z-index: 2;
            min-height: 760px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 5.5rem clamp(2rem, 7vw, 7rem) 7rem;
            max-width: 1100px;
        }

        .hero-eyebrow {
            color: #d6c7a9 !important;
            margin-bottom: 1.05rem;
        }

        .hero-title {
            font-family: "Playfair Display", Georgia, serif;
            font-size: clamp(3.15rem, 5.4vw, 6rem);
            line-height: 0.93;
            letter-spacing: -0.048em;
            color: #f2eee5;
            margin: 0;
        }

        .hero-title em {
            color: #d8c6a5;
            font-style: italic;
        }

        .hero-lede {
            max-width: 760px;
            color: #c3c4bc;
            font-size: 1rem;
            line-height: 1.6;
            margin-top: 1.5rem;
        }

        .hero-actions {
            display: flex;
            gap: 1rem;
            margin-top: 2rem;
        }

        .hero-action-primary,
        .hero-action-secondary {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #45443c;
            padding: 0.78rem 1.35rem;
            font-family: "DM Sans", sans-serif;
            font-size: 0.78rem;
            text-decoration: none !important;
        }

        .hero-action-primary {
            background: #e0d2b8;
            color: #10120f;
        }

        .hero-action-secondary {
            background: rgba(10,12,10,0.30);
            color: #dedbd1;
        }


        @media (max-width: 1000px) {
            .wildid-header {
                grid-template-columns: 1fr;
                row-gap: 0.75rem;
                padding: 0.85rem 1.25rem;
            }

            .wildid-nav {
                justify-content: flex-start;
                flex-wrap: wrap;
                gap: 0.8rem 1.3rem;
            }

            .wildid-status {
                justify-content: flex-start;
            }

            .hero-content {
                padding: 3rem 2rem;
            }

            .hero-title {
                font-size: clamp(2.7rem, 9vw, 4.5rem);
            }
        }
        </style>
        '''.replace("https://images.unsplash.com/photo-1698382439843-ca033a6079c0?fm=jpg&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&ixlib=rb-4.1.0&q=85&w=2600", "https://images.unsplash.com/photo-1698382439843-ca033a6079c0?fm=jpg&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&ixlib=rb-4.1.0&q=85&w=2600"),
        unsafe_allow_html=True,
    )

    st.markdown(
        '''
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
                <div class="step-copy">At or above 0.90 the top candidate is a Match; below it, No Reliable Match.</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '''
            <div class="panel">
                <div class="panel-title">System evidence</div>
                <div style="display:grid;grid-template-columns:1fr auto;gap:0.8rem;border-top:1px solid #2b302b;padding:0.8rem 0;color:#a6a99f;">
                    <span>Model</span><strong style="color:#eee9dc;">MegaDescriptor-T-224</strong>
                    <span>Retrieval</span><strong style="color:#eee9dc;">Cosine similarity</strong>
                    <span>Decision threshold</span><strong style="color:#eee9dc;">0.90</strong>
                    <span>Decision states</span><strong style="color:#eee9dc;">Match / No Reliable Match</strong>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '''
            <div class="panel">
                <div class="panel-title">Prototype evidence · R001 ATRW baseline</div>
                <div class="metric-strip">
                    <div class="metric"><div class="metric-label">TOP-1</div><div class="metric-value">97.75%</div></div>
                    <div class="metric"><div class="metric-label">TOP-5</div><div class="metric-value">99.44%</div></div>
                    <div class="metric"><div class="metric-label">MAP</div><div class="metric-value">94.22%</div></div>
                </div>
                <div style="display:grid;grid-template-columns:1fr auto;gap:0.8rem;padding-top:1rem;color:#a6a99f;">
                    <span>Identities</span><strong style="color:#eee9dc;">107</strong>
                    <span>Gallery images</span><strong style="color:#eee9dc;">642</strong>
                    <span>Query images</span><strong style="color:#eee9dc;">1,245</strong>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )


def render_identify(model, gallery_embeddings, gallery_ids, gallery_filenames):
    st.markdown('<main class="identify-shell">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">01 — query</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title" style="font-size:clamp(2.6rem,4vw,4.6rem);">Identify an individual</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="lede" style="max-width:680px;">Upload a camera-trap image to retrieve the closest known individual candidates.</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Drop a camera-trap image",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )

        st.markdown('<div class="section-label">Demo images</div>', unsafe_allow_html=True)
        demo_a, demo_b = st.columns(2)

        if "selected_demo" not in st.session_state:
            st.session_state.selected_demo = None

        with demo_a:
            if st.button("002253.jpg · MATCH", use_container_width=True):
                st.session_state.selected_demo = "002253.jpg"

        with demo_b:
            if st.button("001454.jpg · REJECT", use_container_width=True):
                st.session_state.selected_demo = "001454.jpg"

        image = None
        image_name = None

        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            image_name = uploaded.name
        elif st.session_state.selected_demo:
            image_name = st.session_state.selected_demo
            image = get_demo_image(image_name)

        if image is not None:
            st.image(image, use_container_width=True)
            if st.button("Identify Individual", key="identify_button"):
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
            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">02 — result</div>
                    <div style="font-family:'Playfair Display',Georgia,serif;font-size:2rem;color:#eee9dc;margin-bottom:0.7rem;">
                        Awaiting a query image.
                    </div>
                    <div class="fineprint" style="font-size:0.9rem;">
                        The result will contain:
                    </div>
                    <div style="margin-top:1.2rem;">
                        <div style="padding:0.9rem 0;border-top:1px solid #2b302b;color:#eee9dc;">01&nbsp;&nbsp; Best candidate <span style="float:right;color:#6f756c;">Closest known individual</span></div>
                        <div style="padding:0.9rem 0;border-top:1px solid #2b302b;color:#eee9dc;">02&nbsp;&nbsp; Similarity score <span style="float:right;color:#6f756c;">Cosine similarity, 0–1</span></div>
                        <div style="padding:0.9rem 0;border-top:1px solid #2b302b;color:#eee9dc;">03&nbsp;&nbsp; Top candidates <span style="float:right;color:#6f756c;">Five ranked gallery identities</span></div>
                        <div style="padding:0.9rem 0;border-top:1px solid #2b302b;color:#eee9dc;">04&nbsp;&nbsp; Evidence confidence <span style="float:right;color:#6f756c;">Decision at threshold 0.90</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            decision = result["decision"]
            if decision == "MATCH":
                st.markdown(
                    f"""
                    <div class="result-match">
                        <div class="decision" style="color:#a2b494;">MATCH</div>
                        <div class="identity">Individual #{result["top1_identity"]}</div>
                        <div class="score">{result["top1_score"]:.2%} similarity</div>
                        <div class="fineprint" style="margin-top:0.6rem;">Strong visual evidence supports this individual match.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-no">
                        <div class="decision" style="color:#c9ad78;">NO RELIABLE MATCH</div>
                        <div class="identity">Evidence below threshold</div>
                        <div class="score">{result["top1_score"]:.2%} similarity</div>
                        <div class="fineprint" style="margin-top:0.6rem;">The available visual evidence is insufficient for a confident individual identification.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown('<div class="panel-title">Top candidates</div>', unsafe_allow_html=True)

            for candidate in result["candidates"]:
                pct = max(0, min(100, candidate["similarity"] * 100))
                st.markdown(
                    f"""
                    <div class="candidate-row">
                        <div class="candidate-rank">0{candidate["rank"]}</div>
                        <div class="candidate-id">#{candidate["identity"]}</div>
                        <div>
                            <div class="candidate-bar">
                                <div class="candidate-bar-fill" style="width:{pct:.2f}%;"></div>
                            </div>
                        </div>
                        <div class="candidate-score">{candidate["similarity"]:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")
    st.markdown(
        """
        <div class="fineprint">
            Prototype threshold: 0.90. This is an R001 evaluation-derived demo rule, not a universal production confidence threshold.
        </div>
        </main>
        """,
        unsafe_allow_html=True,
    )


def render_placeholder(title, eyebrow, body):
    st.markdown('<main class="page-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-title page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="lede page-lede">{body}</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(
        """
        <div class="panel preview-panel">
            <div class="panel-title">Prototype preview</div>
            <div class="fineprint" style="font-size:0.9rem;">
                This layer is intentionally presented as a prototype preview. No unsupported records or conservation claims are generated.
            </div>
        </div>
        </main>
        """,
        unsafe_allow_html=True,
    )


inject_css()

st.markdown(
    '''
    <style>
    .wildid-header { display:flex !important; align-items:center !important; justify-content:space-between !important; gap:2rem !important; min-height:82px !important; height:82px !important; padding:0 4.8vw !important; box-sizing:border-box !important; background:#090b09 !important; }
    .wildid-brand { flex:0 0 auto !important; min-width:340px !important; }
    .wildid-nav { flex:1 1 auto !important; display:flex !important; justify-content:center !important; align-items:center !important; gap:clamp(1.25rem,2vw,2.6rem) !important; overflow:visible !important; min-width:0 !important; }
    .wildid-nav a { flex:0 0 auto !important; display:inline-flex !important; width:auto !important; min-width:max-content !important; color:#eee9dc !important; -webkit-text-fill-color:#eee9dc !important; font-size:.76rem !important; line-height:1 !important; white-space:nowrap !important; padding:.8rem .18rem .72rem !important; }
    .wildid-status { flex:0 0 auto !important; min-width:260px !important; justify-content:flex-end !important; }
    .hero-action-primary, .hero-action-primary:visited { color:#0b0d0b !important; -webkit-text-fill-color:#0b0d0b !important; background:#e0d2b8 !important; border-color:#e0d2b8 !important; }
    .hero-action-secondary, .hero-action-secondary:visited { color:#dedbd1 !important; -webkit-text-fill-color:#dedbd1 !important; background:rgba(10,12,10,.38) !important; border-color:#5a5a50 !important; }
    .hero-action-primary:hover { color:#0b0d0b !important; -webkit-text-fill-color:#0b0d0b !important; background:#eee2c8 !important; }
    .hero-action-secondary:hover { color:#f0ece1 !important; -webkit-text-fill-color:#f0ece1 !important; border-color:#d9c9a9 !important; }
    .page-shell { width:min(1240px, calc(100% - 8rem)) !important; max-width:1240px !important; margin:0 auto !important; padding:5rem 0 7rem !important; box-sizing:border-box !important; }
    .page-title { max-width:920px !important; font-size:clamp(3rem,5.1vw,5.2rem) !important; line-height:.97 !important; overflow-wrap:normal !important; word-break:normal !important; }
    .page-lede { max-width:820px !important; margin-bottom:2rem !important; line-height:1.7 !important; }
    .about-page { padding-top:5.5rem !important; padding-bottom:10rem !important; }
    .about-kicker { max-width:780px !important; }
    .about-intro { display:grid !important; grid-template-columns:minmax(0,1.05fr) minmax(360px,.95fr) !important; gap:clamp(3rem,6vw,7rem) !important; align-items:stretch !important; margin:3rem 0 7rem !important; border-top:1px solid #2b302b !important; border-bottom:1px solid #2b302b !important; }
    .about-intro-copy { padding:4rem 0 4.5rem !important; }
    .about-intro-copy h2 { font-family:"Playfair Display",Georgia,serif !important; font-weight:400 !important; color:#eee9dc !important; font-size:clamp(2.2rem,4vw,4.5rem) !important; line-height:.98 !important; max-width:760px !important; margin:.7rem 0 1.4rem !important; }
    .about-intro-copy p { color:#a3a79d !important; font-family:"DM Sans",sans-serif !important; font-size:1rem !important; line-height:1.8 !important; max-width:720px !important; margin:0 !important; }
    .about-intro-image { min-height:560px !important; position:relative !important; overflow:hidden !important; background:#111410 !important; }
    .about-intro-image::before { content:"" !important; position:absolute !important; inset:0 !important; background-image:url("https://images.unsplash.com/photo-1715088295646-de8244579a3d?auto=format&fit=crop&fm=jpg&ixlib=rb-4.1.0&q=88&w=2200") !important; background-size:cover !important; background-position:center 48% !important; filter:saturate(.62) contrast(1.03) brightness(.76) !important; transform:scale(1.015) !important; }
    .about-intro-image::after { content:"" !important; position:absolute !important; inset:0 !important; background:linear-gradient(90deg,#0b0d0b 0%,rgba(11,13,11,.10) 28%,rgba(11,13,11,.06) 100%),linear-gradient(180deg,rgba(11,13,11,.02),rgba(11,13,11,.55)) !important; }
    .about-image-label { position:absolute !important; z-index:2 !important; left:1.5rem !important; bottom:1.5rem !important; font-family:"DM Mono",monospace !important; font-size:.64rem !important; letter-spacing:.16em !important; color:#d9c9a9 !important; text-transform:uppercase !important; }
    .about-grid { display:grid !important; grid-template-columns:1.1fr .9fr !important; gap:1px !important; background:#2b302b !important; border-top:1px solid #2b302b !important; border-bottom:1px solid #2b302b !important; margin-bottom:6rem !important; }
    .about-card { background:#0b0d0b !important; padding:2.7rem !important; min-height:320px !important; }
    .about-card-wide { grid-row:span 2 !important; min-height:100% !important; }
    .about-card h2, .about-method h2, .about-abstain h2 { font-family:"Playfair Display",Georgia,serif !important; font-weight:400 !important; color:#eee9dc !important; font-size:clamp(2rem,3vw,3.2rem) !important; line-height:1 !important; margin:.7rem 0 1rem !important; }
    .about-card p, .about-method p, .about-abstain p { color:#9da198 !important; font-family:"DM Sans",sans-serif !important; line-height:1.75 !important; font-size:.96rem !important; margin:0 !important; }
    .about-card strong { color:#d9c9a9 !important; }
    .about-method { display:grid !important; grid-template-columns:1fr 1fr !important; gap:4rem !important; padding:0 0 5.5rem !important; border-bottom:1px solid #2b302b !important; margin-bottom:5.5rem !important; }
    .method-list { border-top:1px solid #2b302b !important; }
    .method-list div { display:grid !important; grid-template-columns:45px 1fr auto !important; gap:1rem !important; align-items:center !important; padding:1.15rem 0 !important; border-bottom:1px solid #2b302b !important; }
    .method-list span { font-family:"DM Mono",monospace !important; color:#6f756c !important; font-size:.7rem !important; }
    .method-list strong { color:#eee9dc !important; font-family:"DM Sans",sans-serif !important; font-weight:500 !important; }
    .method-list small { color:#858b81 !important; font-family:"DM Sans",sans-serif !important; }
    .about-evidence { margin-bottom:5.5rem !important; }
    .about-metrics { margin-top:1rem !important; }
    .about-stats { display:flex !important; flex-wrap:wrap !important; gap:2rem !important; padding:1.25rem 0 !important; color:#858b81 !important; font-family:"DM Mono",monospace !important; font-size:.68rem !important; }
    .about-stats b { color:#eee9dc !important; font-size:.9rem !important; }
    .about-note { max-width:820px !important; }
    .about-abstain { display:grid !important; grid-template-columns:1fr 330px !important; gap:4rem !important; align-items:center !important; padding:4.5rem 0 !important; margin-bottom:5.5rem !important; border-top:1px solid #2b302b !important; border-bottom:1px solid #2b302b !important; }
    .abstain-badge { border:1px solid #4b4334 !important; background:#14120e !important; padding:2rem !important; }
    .abstain-badge span, .abstain-badge small { display:block !important; font-family:"DM Mono",monospace !important; color:#857e6c !important; font-size:.64rem !important; letter-spacing:.13em !important; }
    .abstain-badge strong { display:block !important; color:#d9c9a9 !important; font-family:"DM Mono",monospace !important; font-size:.85rem !important; letter-spacing:.1em !important; margin:.7rem 0 !important; }
    .about-roadmap { margin-bottom:4rem !important; }
    .roadmap-grid { display:grid !important; grid-template-columns:repeat(3,1fr) !important; gap:1px !important; background:#2b302b !important; margin-top:1rem !important; }
    .roadmap-grid div { background:#0b0d0b !important; padding:2rem !important; min-height:230px !important; }
    .roadmap-grid span { display:block !important; font-family:"DM Mono",monospace !important; color:#d6c7a9 !important; font-size:.65rem !important; letter-spacing:.15em !important; margin-bottom:2.5rem !important; }
    .roadmap-grid strong { display:block !important; color:#eee9dc !important; font-family:"Playfair Display",Georgia,serif !important; font-size:1.7rem !important; font-weight:400 !important; margin-bottom:.7rem !important; }
    .roadmap-grid p { color:#858b81 !important; line-height:1.55 !important; margin:0 !important; }
    .about-disclaimer { border-left:2px solid #d9c9a9 !important; padding:1.1rem 1.4rem !important; color:#a6a99f !important; background:#10130f !important; line-height:1.65 !important; }
    .about-disclaimer strong { color:#eee9dc !important; }
    .page-shell, .identify-shell { overflow-x:hidden !important; }
    .page-shell *, .identify-shell * { max-width:100%; }
    .page-title, .hero-title { overflow-wrap:break-word !important; }
    .lede { overflow-wrap:break-word !important; }
    @media (max-width:1150px) { .wildid-header{gap:1rem !important;padding:0 3vw !important;} .wildid-brand{min-width:300px !important;} .wildid-brand-sub{display:none !important;} .wildid-nav{gap:1.15rem !important;} .wildid-status{min-width:210px !important;} }
    @media (max-width:900px) { .wildid-header{height:auto !important;min-height:82px !important;flex-wrap:wrap !important;padding:.8rem 5vw !important;} .wildid-brand{min-width:0 !important;} .wildid-nav{order:3 !important;flex-basis:100% !important;justify-content:flex-start !important;overflow-x:auto !important;padding-bottom:.4rem !important;} .wildid-status{display:none !important;} .about-grid,.about-method,.about-abstain,.roadmap-grid,.about-intro{grid-template-columns:1fr !important;} .about-card-wide{grid-row:auto !important;} .about-intro-image{min-height:420px !important;} .page-shell,.identify-shell{width:100% !important;padding-left:6vw !important;padding-right:6vw !important;} }
    </style>
    ''',
    unsafe_allow_html=True,
)

valid_pages = {"Overview", "Identify", "Individuals", "Sightings", "About"}
query_page = st.query_params.get("page")
if query_page in valid_pages:
    st.session_state.page = query_page
elif "page" not in st.session_state:
    st.session_state.page = "Overview"

render_brandbar()

page = st.session_state.page

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
    render_placeholder(
        "Individual profiles",
        "02 — individual intelligence",
        "A future layer for persistent individual records, evidence, and observation context.",
    )

elif page == "Sightings":
    render_placeholder(
        "Sighting history",
        "03 — observation log",
        "Confirmed matches can become dated observations in the full WildID system. No sighting records are fabricated in this prototype.",
    )

elif page == "About":
    st.markdown('<main class="page-shell about-page">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">About WildID · Project brief</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title page-title">Individual wildlife recognition, built around evidence.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="lede page-lede about-kicker">
            WildID addresses the gap between <strong>detecting an animal</strong> and determining whether the animal
            has been seen before. The prototype compares a camera-trap image with known individuals, ranks visual
            candidates, and can return <strong>No Reliable Match</strong> instead of forcing an identity.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="about-intro">
            <div class="about-intro-copy">
                <div class="panel-title">01 · Why this matters</div>
                <h2>From “a tiger was detected” to “which tiger is this?”</h2>
                <p>Wildlife monitoring produces large volumes of camera-trap images, but identifying the same individual across observations is a different problem from species detection. WildID is designed around that individual-level question. It turns a query image into a visual embedding, compares it against a gallery of known individuals, ranks the closest candidates, and exposes the evidence used for the decision.</p>
            </div>
            <div class="about-intro-image"><div class="about-image-label">Tiger · wildlife reference image</div></div>
        </section>

        <section class="about-grid">
            <article class="about-card about-card-wide">
                <div class="panel-title">02 · Core distinction</div>
                <h2>Detection is not identity.</h2>
                <p>A detector can establish that an animal appears in a frame. Re-identification asks a harder question: does this animal correspond to one of the known individuals already represented in the gallery? WildID keeps those concepts separate. The system does not treat the nearest visual match as automatically correct; it ranks candidates and uses a confidence rule to decide whether the evidence is strong enough for the prototype decision.</p>
                <br>
                <p>That evidence-first design is important for conservation workflows because an incorrect individual assignment can contaminate later records. A valid system therefore needs an explicit way to say <strong>No Reliable Match</strong> when the available evidence is insufficient.</p>
            </article>
            <article class="about-card"><div class="panel-title">03 · What works today</div><h2>Tiger-first and deliberately small.</h2><p>The current implementation is a <strong>small working tiger re-identification prototype</strong>. It demonstrates the actual retrieval pipeline rather than claiming a finished conservation platform.</p></article>
            <article class="about-card"><div class="panel-title">04 · What it is not</div><h2>Not a finished deployment.</h2><p>Individual profiles, longitudinal sighting history, spatial intelligence, live camera integration, and broad multi-species validation are not being presented as completed features. They are the next system layers to build and validate.</p></article>
        </section>

        <section class="about-method">
            <div><div class="panel-title">05 · Working methodology</div><h2>Query → Retrieve → Decide</h2><p>A query image is converted into an embedding using MegaDescriptor-T-224. That embedding is compared with gallery embeddings using cosine similarity. The prototype ranks the strongest candidates and applies the evaluation-derived 0.90 rule used for the current demo decision.</p><br><p>This is a retrieval prototype, not a claim that 0.90 is a universal production confidence threshold. Thresholds would need species-specific validation and field testing before deployment.</p></div>
            <div class="method-list"><div><span>01</span><strong>Query</strong><small>Camera-trap image</small></div><div><span>02</span><strong>Embed</strong><small>Visual representation</small></div><div><span>03</span><strong>Retrieve</strong><small>Rank known individuals</small></div><div><span>04</span><strong>Decide</strong><small>Match or No Reliable Match</small></div></div>
        </section>

        <section class="about-evidence">
            <div class="panel-title">06 · R001 ATRW baseline · measured prototype evidence</div>
            <div class="metric-strip about-metrics"><div class="metric"><div class="metric-label">TOP-1</div><div class="metric-value">97.75%</div></div><div class="metric"><div class="metric-label">TOP-5</div><div class="metric-value">99.44%</div></div><div class="metric"><div class="metric-label">MAP</div><div class="metric-value">94.22%</div></div></div>
            <div class="about-stats"><span><b>107</b> identities</span><span><b>642</b> gallery images</span><span><b>1,245</b> query images</span><span><b>0.90</b> demo threshold</span></div>
            <p class="fineprint about-note">These are measured R001 offline evaluation results on the ATRW split used for this prototype. They are evidence that the current retrieval prototype works on the evaluated split, not live production accuracy and not proof of deployment readiness.</p>
        </section>

        <section class="about-abstain"><div class="about-abstain-copy"><div class="panel-title">07 · Evidence over forced answers</div><h2>“No Reliable Match” is a feature, not a failure.</h2><p>If the strongest candidate does not meet the prototype rule, WildID does not invent certainty. This creates a safer decision boundary for a future conservation system and makes uncertainty visible to the user instead of hiding it behind a forced identity.</p></div><div class="abstain-badge"><span>DECISION STATE</span><strong>NO RELIABLE MATCH</strong><small>Evidence below threshold</small></div></section>

        <section class="about-roadmap"><div class="panel-title">08 · Expansion beyond the current prototype</div><div class="roadmap-grid"><div><span>NOW</span><strong>Working tiger Re-ID</strong><p>Real image retrieval, ranked candidates, measured R001 evaluation, and an explicit abstention state.</p></div><div><span>NEXT</span><strong>Other relevant animals</strong><p>The architecture is intended to extend to other relevant wildlife species, with species-appropriate datasets and validation.</p></div><div><span>FUTURE</span><strong>Conservation intelligence</strong><p>Validated individual profiles, sighting history, spatial context, and integration with wildlife monitoring workflows.</p></div></div></section>

        <section class="about-disclaimer"><strong>Prototype scope:</strong> WildID is currently a small, working tiger re-identification prototype. The broader multi-species conservation intelligence system is the intended direction. Other relevant animals can be supported as suitable datasets, models, and validation are added; those future layers are not being represented here as already deployed.</section>
        </main>
    """,
        unsafe_allow_html=True,
    )

