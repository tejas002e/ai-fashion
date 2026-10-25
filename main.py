import hashlib
import logging

import streamlit as st
import cv2
import numpy as np

from portrait_analysis import (
    MAX_UPLOAD_BYTES,
    PortraitAnalysisError,
    analyze_portrait,
    decode_uploaded_image,
)

# Set page configuration
st.set_page_config(
    page_title="Chroma Fashion AI | Personal Color Analysis",
    page_icon="✨",
    layout="wide"
)

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_status' not in st.session_state:
    st.session_state.analysis_status = 'idle'
if 'portrait_upload_signature' not in st.session_state:
    # Start clean when upgrading a session that used the earlier analyzer state.
    st.session_state.portrait_upload_signature = None
    st.session_state.portrait_rgb = None
    st.session_state.analysis_results = None
    st.session_state.analysis_status = 'idle'
    st.session_state.error_message = None
    st.session_state.portrait_upload_version = 0
if 'portrait_upload_version' not in st.session_state:
    st.session_state.portrait_upload_version = 0


def reset_portrait_analysis():
    """Clear analysis state and rotate the uploader key without mutating its value."""
    st.session_state.portrait_upload_version += 1
    st.session_state.portrait_upload_signature = None
    st.session_state.portrait_rgb = None
    st.session_state.analysis_results = None
    st.session_state.analysis_status = 'idle'
    st.session_state.error_message = None

# Custom CSS for Myntra/H&M Aesthetic with Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --bg-dark: #0a0a0c;
        --sidebar-bg: #0d0d10;
        --surface-raised: #15151b;
        --card-bg: rgba(255, 255, 255, 0.05);
        --soft-bg: rgba(255, 255, 255, 0.035);
        --glass-border: rgba(255, 255, 255, 0.1);
        --primary-accent: #6366f1;
        --secondary-accent: #a855f7;
        --primary-accent-rgb: 99, 102, 241;
        --secondary-accent-rgb: 168, 85, 247;
        --text-main: #f8fafc;
        --text-muted: #b3bdcc;
        --focus-ring: #c4b5fd;
    }

    :root { color-scheme: dark; }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
        background-color: var(--bg-dark) !important;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: var(--text-main);
        letter-spacing: -0.025em;
    }
    [data-testid="stCaptionContainer"] { color: var(--text-muted); }

    /* Subtle motion; reduced-motion preferences are honored below. */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { transform: translateY(12px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    
    .fade-in { animation: fadeIn 0.45s ease-out both; }
    .slide-up { animation: slideUp 0.4s ease-out both; }

    /* Navigation stays in document flow to avoid covering content. */
    .st-key-top-navigation {
        position: sticky;
        top: 0;
        z-index: 20;
        margin-bottom: 1.25rem;
        padding: 0.65rem 0.9rem;
        background: rgba(10, 10, 12, 0.92);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid var(--glass-border);
        border-radius: 0 0 16px 16px;
    }
    .brand-logo {
        display: flex;
        flex-direction: column;
        line-height: 1.05;
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: -0.06em;
        white-space: nowrap;
    }
    .brand-name {
        color: var(--focus-ring);
        -webkit-text-fill-color: var(--focus-ring);
    }
    .brand-logo .brand-descriptor {
        margin-top: 0.25rem;
        font-size: 0.58rem;
        letter-spacing: 0.24em;
        color: var(--text-muted);
    }
    .st-key-top-navigation [data-testid="stRadio"] { margin: 0; }
    .st-key-top-navigation div[role="radiogroup"] {
        display: flex;
        flex-flow: row wrap;
        justify-content: flex-end;
        gap: 0.35rem;
    }
    .st-key-top-navigation div[role="radiogroup"] > label {
        margin: 0;
        padding: 0.45rem 0.7rem;
        border: 1px solid transparent;
        border-radius: 999px;
        color: var(--text-muted);
        font-weight: 650;
        font-size: 0.82rem;
        letter-spacing: 0.045em;
        text-transform: uppercase;
        transition: color 0.18s ease, background 0.18s ease, border-color 0.18s ease;
    }
    .st-key-top-navigation div[role="radiogroup"] > label:hover {
        color: var(--text-main);
        background: var(--soft-bg);
    }
    .st-key-top-navigation div[role="radiogroup"] > label:has(input:checked) {
        color: var(--text-main);
        background: rgba(var(--primary-accent-rgb), 0.18);
        border-color: rgba(var(--primary-accent-rgb), 0.55);
    }
    .st-key-top-navigation input[type="radio"] {
        accent-color: var(--primary-accent);
    }

    /* Sidebar utility panel */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--glass-border);
    }

    .st-key-sidebar-reset-card,
    .st-key-sidebar-tip-card {
        background: var(--card-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .st-key-sidebar-reset-card { padding: 0.55rem; }

    /* This app uses one in-page navigation group rather than multipage links. */
    [data-testid="stSidebarNav"] { display: none !important; }

    /* Hero Section */
    .hero-section {
        background:
            radial-gradient(circle at 88% 10%, rgba(var(--primary-accent-rgb), 0.18), transparent 42%),
            radial-gradient(circle at 8% 90%, rgba(var(--secondary-accent-rgb), 0.16), transparent 44%),
            var(--soft-bg);
        border: 1px solid var(--glass-border);
        padding: clamp(2.25rem, 6vw, 5rem) clamp(1.25rem, 4vw, 3.5rem);
        border-radius: 28px;
        margin-bottom: clamp(2rem, 4vw, 3.5rem);
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
    }
    .hero-title {
        max-width: 1000px;
        margin: 0 auto 1rem;
        font-size: clamp(2.2rem, 6.5vw, 4.5rem);
        font-weight: 800;
        color: var(--text-main);
        line-height: 1.05;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: clamp(1rem, 2vw, 1.2rem);
        color: var(--text-muted);
        margin: 0 auto;
        max-width: 620px;
        line-height: 1.65;
    }

    /* Consistent buttons and visible keyboard focus */
    .stButton>button {
        background: linear-gradient(to right, var(--primary-accent), var(--secondary-accent)) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        padding: 0.72rem 1.15rem !important;
        min-height: 44px;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
        box-shadow: 0 4px 15px rgba(var(--primary-accent-rgb), 0.24) !important;
    }
    .stButton>button:hover:not(:disabled) {
        transform: translateY(-1px) !important;
        filter: brightness(1.08);
        box-shadow: 0 7px 20px rgba(var(--primary-accent-rgb), 0.3) !important;
    }
    .stButton>button:focus-visible,
    a:focus-visible,
    input:focus-visible,
    textarea:focus-visible,
    select:focus-visible,
    .st-key-top-navigation label:has(input:focus-visible) {
        outline: 3px solid var(--focus-ring) !important;
        outline-offset: 3px !important;
    }

    /* Glass and product cards */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(var(--primary-accent-rgb), 0.42);
        transform: translateY(-2px);
    }
    .product-card {
        background: var(--card-bg);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        overflow: hidden;
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
        backdrop-filter: blur(12px);
        height: 100%;
    }
    .product-card:hover {
        transform: translateY(-3px);
        border-color: rgba(var(--primary-accent-rgb), 0.48);
        box-shadow: 0 14px 28px rgba(0,0,0,0.3);
    }
    .product-img {
        position: relative;
        width: 100%;
        height: clamp(190px, 25vw, 300px);
        overflow: hidden;
        background: var(--surface-raised);
    }
    .product-img img {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center;
        transition: transform 0.28s ease;
    }
    .product-card:hover .product-img img { transform: scale(1.025); }
    .product-info { padding: 1.5rem; }
    .product-brand {
        color: #a5a7ff;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .product-name { color: var(--text-main); font-size: 1rem; font-weight: 650; margin-bottom: 0.5rem; }
    .product-price { color: var(--text-muted); font-size: 0.9rem; }

    /* About feature cards */
    .feature-card {
        background: var(--soft-bg);
        border: 1px solid var(--glass-border);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover { transform: translateY(-3px); border-color: rgba(var(--primary-accent-rgb), 0.4); }
    .feature-icon { font-size: 2.5rem; margin-bottom: 1rem; }

    .swatch {
        position: relative;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-block;
        border: 2px solid var(--glass-border);
        transition: transform 0.18s ease;
    }
    .swatch:hover { transform: scale(1.08); }
    .swatch-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1rem;
    }
    .app-footer {
        margin-top: 3rem;
        padding: 1.25rem 0 0.5rem;
        border-top: 1px solid var(--glass-border);
        color: var(--text-muted);
        font-size: 0.78rem;
        line-height: 1.6;
        text-align: center;
    }
    [data-testid="stAlert"] { border-radius: 14px; }

    @media (max-width: 900px) {
        .hero-section { border-radius: 24px; }
        .st-key-top-navigation > div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
            align-items: stretch;
            gap: 0.5rem;
        }
        .st-key-top-navigation div[role="radiogroup"] { justify-content: center; }
        .st-key-home-trends [data-testid="stHorizontalBlock"],
        .st-key-outfit-suggestions [data-testid="stHorizontalBlock"],
        .st-key-about-cards [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        .st-key-portrait-and-quality > div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        .st-key-portrait-and-quality [data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
        }
        .st-key-home-trends [data-testid="column"],
        .st-key-outfit-suggestions [data-testid="column"],
        .st-key-about-cards [data-testid="column"] {
            flex: 1 1 calc(50% - 0.75rem) !important;
            width: calc(50% - 0.75rem) !important;
            min-width: 0 !important;
        }
    }
    @media (max-width: 560px) {
        [data-testid="stMainBlockContainer"] {
            padding: 0.65rem 0.8rem 2rem;
        }
        .st-key-top-navigation { padding: 0.55rem 0.7rem; }
        .st-key-top-navigation .brand-logo {
            flex-direction: row;
            align-items: baseline;
            justify-content: center;
            gap: 0.45rem;
            text-align: center;
        }
        .st-key-top-navigation .brand-descriptor {
            margin-top: 0;
            font-size: 0.56rem;
            letter-spacing: 0.16em;
        }
        .st-key-top-navigation div[role="radiogroup"] { justify-content: stretch; gap: 0.3rem; }
        .st-key-top-navigation div[role="radiogroup"] > label {
            flex: 1 1 calc(50% - 0.3rem);
            justify-content: center;
            padding: 0.5rem 0.35rem;
            font-size: 0.82rem;
        }
        .hero-section { padding: 2rem 1rem; border-radius: 18px; }
        .hero-title { font-size: clamp(2rem, 9vw, 2.65rem); }
        .hero-subtitle { font-size: 0.98rem; }
        .st-key-home-trends [data-testid="column"],
        .st-key-outfit-suggestions [data-testid="column"],
        .st-key-about-cards [data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
        }
        .st-key-dominant-color-row > div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        .st-key-dominant-color-row [data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
        }
        .product-img { height: min(82vw, 320px); }
        .product-info { padding: 1.1rem; }
        .feature-card { padding: 1.35rem; }
        .stButton>button { padding-inline: 0.85rem !important; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- TOP NAVIGATION ---
NAVIGATION_PAGES = {
    "Home": "Home",
    "Analyzer": "Style Analyzer",
    "Guide": "Color Guide",
    "About": "About",
}
PAGE_NAVIGATION = {page: label for label, page in NAVIGATION_PAGES.items()}


def sync_page_from_navigation():
    st.session_state.page = NAVIGATION_PAGES[st.session_state.primary_navigation]


if st.session_state.page not in PAGE_NAVIGATION:
    st.session_state.page = "Home"
if "primary_navigation" not in st.session_state:
    st.session_state.primary_navigation = PAGE_NAVIGATION[st.session_state.page]

with st.container(key="top-navigation"):
    brand_col, nav_col = st.columns(
        [1.1, 4],
        vertical_alignment="center",
        gap="small",
    )
    with brand_col:
        st.markdown(
            '<div class="brand-logo" aria-label="Chroma Fashion AI">'
            '<span class="brand-name">CHROMA</span>'
            '<span class="brand-descriptor">FASHION AI</span></div>',
            unsafe_allow_html=True,
        )
    with nav_col:
        st.radio(
            "Primary navigation",
            options=list(NAVIGATION_PAGES),
            horizontal=True,
            key="primary_navigation",
            on_change=sync_page_from_navigation,
            label_visibility="collapsed",
            width="stretch",
        )

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛠️ CHROMA STYLIST")
    st.caption("Computer-vision color guidance")
    st.divider()

    with st.container(key="sidebar-reset-card"):
        if st.button("✨ Reset Stylist", width="stretch"):
            reset_portrait_analysis()
            st.rerun()

    with st.container(key="sidebar-tip-card"):
        st.markdown("#### 💡 PRO TIP")
        st.caption(
            "Use a clear, front-facing portrait in even natural light. "
            "Lighting, white balance, and makeup can affect estimates."
        )

    st.markdown("### 📊 SESSION")
    st.caption(f"Active: {st.session_state.page}")

# --- RULE-BASED STYLE RECOMMENDATIONS ---
def get_recommendations(tone, undertone):
    # Curated palettes are styling suggestions, not classifier training data.
    palettes = {
        ("Light", "Cool"): {"colors": ["#87CEEB", "#E6E6FA", "#FFB6C1", "#40E0D0", "#000080"], "names": ["Sky Blue", "Lavender", "Pastel Pink", "Mint", "Navy"]},
        ("Light", "Warm"): {"colors": ["#FFD700", "#FF7F50", "#98FB98", "#F5F5DC", "#8B4513"], "names": ["Gold", "Coral", "Spring Green", "Cream", "Sienna"]},
        ("Light", "Neutral"): {"colors": ["#D8C6C1", "#8FBCB0", "#6B8E9E", "#F5F5F0", "#A68A7B"], "names": ["Dusty Rose", "Soft Sage", "Lagoon Blue", "Ivory", "Taupe"]},
        ("Medium", "Cool"): {"colors": ["#4169E1", "#800080", "#C71585", "#00CED1", "#666666"], "names": ["Royal Blue", "Deep Plum", "Magenta", "Ocean Teal", "Slate"]},
        ("Medium", "Warm"): {"colors": ["#808000", "#DAA520", "#D2691E", "#FF4500", "#4b3621"], "names": ["Olive", "Mustard", "Chocolate", "Terracotta", "Espresso"]},
        ("Medium", "Neutral"): {"colors": ["#967BB6", "#739B8E", "#B76E79", "#54717A", "#B7A99A"], "names": ["Muted Plum", "Eucalyptus", "Dusty Rose", "Blue Slate", "Stone"]},
        ("Dark", "Cool"): {"colors": ["#0000FF", "#4B0082", "#FF00FF", "#FFFFFF", "#000000"], "names": ["Electric Blue", "Indigo", "Fuchsia", "Optic White", "Jet Black"]},
        ("Dark", "Warm"): {"colors": ["#FFD700", "#B8860B", "#DC143C", "#556B2F", "#3d0c02"], "names": ["Pure Gold", "Bronze", "Ruby", "Deep Olive", "Coffee"]},
        ("Dark", "Neutral"): {"colors": ["#D8C6C1", "#70A894", "#A889B7", "#F5F5F0", "#34454D"], "names": ["Dusty Rose", "Jade", "Muted Violet", "Soft White", "Deep Slate"]}
    }
    return palettes.get((tone, undertone), {"colors": ["#282c3f", "#7e818c"], "names": ["Charcoal", "Slate"]})

# --- PAGE CONTENT ---

if st.session_state.page == 'Home':
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title slide-up">Find Colors That<br>Match Your Aura</h1>
        <p class="hero-subtitle slide-up">A portrait-based color estimate and a practical palette to explore.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container(key="home-trends"):
        st.markdown("### TRENDING THIS SEASON")
        cols = st.columns(4, gap="medium")
        trends = [
            {
                "img": "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=400",
                "alt": "Model wearing a vibrant summer outfit",
                "name": "Vibrant Summer Set",
                "price": "₹1,999",
            },
            {
                "img": "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?auto=format&fit=crop&w=400",
                "alt": "Neutral linen outfit displayed on a model",
                "name": "Minimalist Linen",
                "price": "₹2,499",
            },
            {
                "img": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f?auto=format&fit=crop&w=400",
                "alt": "Relaxed streetwear look in a light neutral palette",
                "name": "Street Chic Oversize",
                "price": "₹1,299",
            },
            {
                "img": "https://images.unsplash.com/photo-1554412933-514a83d2f3c8?auto=format&fit=crop&w=400",
                "alt": "Formal evening outfit in a deep jewel tone",
                "name": "Royal Evening Gown",
                "price": "₹4,999",
            },
        ]
        for i, t in enumerate(trends):
            with cols[i]:
                st.markdown(f"""
                <article class="product-card slide-up" style="animation-delay: {i*0.08}s">
                    <div class="product-img">
                        <img src="{t['img']}" alt="{t['alt']}" loading="lazy" decoding="async">
                    </div>
                    <div class="product-info">
                        <div class="product-brand">SEASONAL DROP</div>
                        <div class="product-name">{t['name']}</div>
                        <div class="product-price">{t['price']}</div>
                    </div>
                </article>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Style Analyzer':
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("## STYLE ANALYZER")
    st.caption(
        "Upload a clear, front-facing portrait. The analyzer samples several "
        "facial regions and turns the measured color into a styling estimate."
    )
    upload_key = f"portrait_upload_{st.session_state.portrait_upload_version}"
    uploaded_file = st.file_uploader(
        "DRAG AND DROP YOUR PORTRAIT",
        type=["jpg", "jpeg", "png"],
        key=upload_key,
        help="JPEG or PNG, up to 10 MB. Images are normalized and large images are resized before analysis.",
    )

    if uploaded_file is not None:
        file_size = int(uploaded_file.size)
        if file_size > MAX_UPLOAD_BYTES:
            upload_signature = f"oversized:{uploaded_file.name}:{file_size}"
            upload_bytes = None
            upload_error = "The image is larger than the 10 MB upload limit. Choose a smaller file."
        else:
            upload_bytes = uploaded_file.getvalue()
            upload_signature = hashlib.sha256(upload_bytes).hexdigest()
            upload_error = None

        if upload_signature != st.session_state.portrait_upload_signature:
            # Clear old output before validating the new file so it can never
            # appear alongside an invalid upload or a failed analysis.
            st.session_state.portrait_upload_signature = upload_signature
            st.session_state.portrait_rgb = None
            st.session_state.analysis_results = None
            st.session_state.analysis_status = "idle"
            st.session_state.error_message = None

            if upload_error:
                st.session_state.analysis_status = "error"
                st.session_state.error_message = upload_error
            else:
                try:
                    normalized_image = decode_uploaded_image(upload_bytes)
                    st.session_state.portrait_rgb = np.asarray(
                        normalized_image,
                        dtype=np.uint8,
                    )
                except PortraitAnalysisError as exc:
                    st.session_state.analysis_status = "error"
                    st.session_state.error_message = str(exc)
    elif st.session_state.portrait_upload_signature is not None:
        # The user removed the selected file in the uploader.
        reset_portrait_analysis()

    portrait_rgb = st.session_state.portrait_rgb
    analysis_status = st.session_state.analysis_status

    if portrait_rgb is None:
        if analysis_status == "error":
            st.error(st.session_state.error_message)
            st.button(
                "CHOOSE ANOTHER IMAGE",
                on_click=reset_portrait_analysis,
                width="stretch",
            )
        else:
            st.info("Upload a portrait to begin. Use one clearly visible face.")
    else:
        if analysis_status == "idle":
            st.image(
                portrait_rgb,
                caption="Uploaded portrait preview before face detection.",
                width="stretch",
            )
            if st.button("ANALYZE PORTRAIT", width="stretch"):
                progress_bar = st.progress(0, text="Starting portrait analysis…")

                def update_analysis_progress(value, message):
                    progress_bar.progress(value, text=message)

                try:
                    result = analyze_portrait(
                        portrait_rgb,
                        progress_callback=update_analysis_progress,
                    )
                    x, y, width, height = result["face_rect"]
                    annotated_image = portrait_rgb.copy()
                    box_thickness = max(2, round(min(annotated_image.shape[:2]) / 500))
                    cv2.rectangle(
                        annotated_image,
                        (x, y),
                        (x + width, y + height),
                        (99, 102, 241),
                        box_thickness,
                    )
                    result["annotated_image"] = annotated_image
                    st.session_state.analysis_results = result
                    st.session_state.analysis_status = "complete"
                    st.session_state.error_message = None
                except PortraitAnalysisError as exc:
                    st.session_state.analysis_results = None
                    st.session_state.analysis_status = "error"
                    st.session_state.error_message = str(exc)
                except Exception:
                    logging.exception("Unexpected portrait analysis failure")
                    st.session_state.analysis_results = None
                    st.session_state.analysis_status = "error"
                    st.session_state.error_message = (
                        "The portrait could not be analyzed safely. "
                        "Try a clearer, front-facing image."
                    )
                finally:
                    progress_bar.empty()
                st.rerun()

        if analysis_status == "error":
            st.error(st.session_state.error_message)
            st.button(
                "CHOOSE ANOTHER IMAGE",
                on_click=reset_portrait_analysis,
                width="stretch",
            )

        elif analysis_status == "complete" and st.session_state.analysis_results:
            res = st.session_state.analysis_results
            rec = get_recommendations(res["tone"], res["undertone"])
            with st.container(key="analysis-results"):
                st.markdown("### DETECTED FACE")
                with st.container(key="portrait-and-quality"):
                    portrait_col, quality_col = st.columns(
                        [1.15, 0.85],
                        gap="large",
                        vertical_alignment="top",
                    )
                    with portrait_col:
                        st.image(
                            res["annotated_image"],
                            width="stretch",
                            caption=(
                                "Uploaded portrait with the selected face outlined "
                                "in purple."
                            ),
                        )
                    with quality_col:
                        st.markdown("#### PHOTO QUALITY")
                        if res["quality_warnings"]:
                            st.warning(
                                "Notes to consider:\n\n"
                                + "\n\n".join(
                                    f"- {note}" for note in res["quality_warnings"]
                                )
                            )
                        else:
                            st.success("No major capture issues detected.")
                        if res["multiple_faces"]:
                            st.info(
                                f"{res['face_count']} usable faces were found. "
                                "The largest face was selected for this estimate."
                            )
                        st.caption(
                            "Lighting, white balance, makeup, and image quality "
                            "can change the measured color and inferred categories."
                        )

                st.markdown("---")
                st.markdown("### YOUR ESTIMATED PROFILE")
                tone_col, undertone_col = st.columns(2, gap="medium")
                with tone_col:
                    st.metric("Skin tone estimate", res["tone"])
                with undertone_col:
                    st.metric("Undertone estimate", res["undertone"])
                st.caption(
                    "These are broad, rule-based photo estimates—not a validated "
                    "skin-tone scale or a substitute for professional advice."
                )

                st.markdown("### DOMINANT EXTRACTED COLOUR")
                with st.container(key="dominant-color-row"):
                    color_col, measurement_col = st.columns(
                        [0.45, 1.55],
                        gap="medium",
                        vertical_alignment="center",
                    )
                    with color_col:
                        st.color_picker(
                            "Representative sampled color",
                            value=res["color_hex"],
                            disabled=True,
                        )
                    with measurement_col:
                        red, green, blue = res["dominant_color"]
                        st.markdown(
                            f"**{res['color_hex']}**  \n"
                            f"Measured RGB: {red}, {green}, {blue}"
                        )
                        lightness, a_axis, b_axis = res["lab"]
                        st.caption(
                            f"CIELAB: L* {lightness:.1f}, a* {a_axis:.1f}, "
                            f"b* {b_axis:.1f}. Median from {res['cluster_pixels']:,} "
                            f"pixels in the largest Lab cluster "
                            f"({res['sampled_pixels']:,} sampled across forehead "
                            "and cheeks)."
                        )

                st.markdown("### RECOMMENDED PALETTE")
                swatches = "".join(
                    (
                        f'<span class="swatch" role="img" '
                        f'aria-label="{name} ({color})" title="{name} ({color})" '
                        f'style="background-color:{color};"></span>'
                    )
                    for name, color in zip(rec["names"], rec["colors"])
                )
                st.markdown(
                    f'<div class="swatch-container">{swatches}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(" · ".join(
                    f"{name} {color}" for name, color in zip(rec["names"], rec["colors"])
                ))

                st.markdown("---")
                st.markdown("### OUTFIT SUGGESTIONS")
                st.caption(
                    "Visual inspiration matched to the suggested palette; these "
                    "examples are not live retailer listings."
                )
                with st.container(key="outfit-suggestions"):
                    outfit_cols = st.columns(3, gap="medium")
                    outfits = [
                        {
                            "name": f"{rec['names'][0]} casual shirt",
                            "detail": "A relaxed base layer",
                            "img": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=400",
                            "alt": "Casual cotton shirt styled as an everyday outfit",
                        },
                        {
                            "name": f"{rec['names'][1]} cotton tee",
                            "detail": "An easy everyday pairing",
                            "img": "https://images.unsplash.com/photo-1581655353564-df123a1eb820?auto=format&fit=crop&w=400",
                            "alt": "Cotton tee styled with casual trousers",
                        },
                        {
                            "name": f"{rec['names'][2]} summer layer",
                            "detail": "A soft accent for the palette",
                            "img": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?auto=format&fit=crop&w=400&q=80",
                            "alt": "Lightweight jacket shown as a color accent",
                        },
                    ]
                    for i, outfit in enumerate(outfits):
                        with outfit_cols[i]:
                            st.markdown(f"""
                            <article class="product-card slide-up">
                                <div class="product-img">
                                    <img src="{outfit['img']}" alt="{outfit['alt']}" loading="lazy" decoding="async">
                                </div>
                                <div class="product-info">
                                    <div class="product-brand">PALETTE MATCH</div>
                                    <div class="product-name">{outfit['name']}</div>
                                    <div class="product-price">{outfit['detail']}</div>
                                </div>
                            </article>
                            """, unsafe_allow_html=True)

            st.button(
                "ANALYZE ANOTHER PORTRAIT",
                on_click=reset_portrait_analysis,
                width="stretch",
            )

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == 'Color Guide':
    st.title("Style Guide: Undertone & Color Palette")
    
    st.write("Understand your undertones and find your signature style. Below are guidance notes for different skin undertones.")
    
    # 1. Cool Undertone Section
    st.header("Cool Undertones")
    st.write("Individuals with cool undertones typically have hints of pink, red, or blue in their skin. Silver jewelry and cool-toned colors like blues, purples, and blacks tend to look best. These shades complement the natural cool pigments in your skin, creating a balanced and harmonious appearance.")
    
    st.divider()
    
    # 2. Warm Undertone Section
    st.header("Warm Undertones")
    st.write("Those with warm undertones often have yellow, peachy, or golden hues. Gold jewelry and earthy colors like beige, brown, and olive green are highly flattering. These warm tones enhance the natural radiance of your skin, making it appear vibrant and healthy.")
    
    st.divider()
    
    # 3. Neutral Undertone Section
    st.header("Neutral Undertones")
    st.write("Neutral undertones have a mix of warm and cool pigments, meaning most colors will look great. You have the flexibility to wear both gold and silver jewelry. Soft, muted colors like dusty rose, jade green, and lagoon blue often look particularly elegant.")
    
    # Analysis Summary if results exist
    user_results = st.session_state.get('analysis_results')
    if user_results:
        st.info(
            f"Current profile: the portrait analysis estimates a "
            f"{user_results['undertone'].lower()} undertone. Lighting, white balance, "
            "makeup, and image quality can affect this estimate."
        )

elif st.session_state.page == 'About':
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.markdown("## ABOUT THE TECHNOLOGY")
    with st.container(key="about-cards"):
        cols = st.columns(3, gap="medium")
        features = [
            {"icon": "🎨", "title": "COLOR ESTIMATES", "desc": "Lab-space clustering samples forehead and cheek regions."},
            {"icon": "👁️", "title": "FACE DETECTION", "desc": "OpenCV Haar Cascades select the largest usable face."},
            {"icon": "👔", "title": "PERSONAL STYLING", "desc": "Rule-based palettes turn measured color into suggestions."}
        ]
        for i, f in enumerate(features):
            with cols[i]:
                st.markdown(f"""
                <article class="feature-card slide-up" aria-label="{f['title']}">
                    <div class="feature-icon" aria-hidden="true">{f['icon']}</div>
                    <div style="font-weight: 700; font-size: 0.9rem; margin-bottom: 4px; color: var(--text-main);">{f['title']}</div>
                    <div style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.55; margin-bottom: 8px;">{f['desc']}</div>
                </article>
                """, unsafe_allow_html=True)
    
    st.markdown("### FUTURE SCOPE")
    st.info("Coming Soon: Virtual Try-On, Real-time Webcam Analysis, and Integration with H&M/Myntra APIs for live shopping.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
    <footer class="app-footer" role="contentinfo">
        © 2026 CHROMA FASHION AI. ALL RIGHTS RESERVED. <br>
        POWERED BY OPENCV COMPUTER VISION & RULE-BASED COLOR GUIDANCE.
    </footer>
""", unsafe_allow_html=True)
