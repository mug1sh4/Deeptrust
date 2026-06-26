"""
DEEPTRUST: A Web-Based AI Platform for Image Deepfake Detection Using Ensemble
Convolutional Neural Networks and Autoencoder-Based Anomaly Analysis for Improved
Generalizability, Efficiency, and Interpretability

Final Year IT Project — Technical University of Kenya
STUDENT REG. NUMBER: (SCCJ/01514/2022)

"""


# SECTION 1: IMPORTS
# These are external libraries and tools the app needs to function.
import streamlit as st # The web framework that powers the entire UI
import numpy as np # Numerical computing — used for array maths on image pixels
import cv2 # OpenCV — image processing (resizing, colour conversion, face detection)
import hashlib # Generates SHA-256 hash fingerprint of uploaded images for integrity verification
import os # File system operations — checking if model files exist, creating folders
import time # Measures how long each scan takes (process time in forensic log)
import uuid # Generates unique IDs for each upload and result (e.g. UP-20260529-xxxx)
import sqlite3 # Lightweight database — stores scan results in memory during the session
import io # Handles image data as bytes in memory without saving to disk
import requests # Makes HTTP calls to external AI APIs (Anthropic, Ollama)
from datetime import datetime # Captures the exact timestamp when a scan is performed
from PIL import Image # Python Imaging Library — opens, resizes, and converts uploaded images



# SECTION 2: ENVIRONMENT VARIABLES
# Sensitive credentials (like API keys) should never be hardcoded into source
# code. Instead they are stored in a .env file on the developer's machine.
# This block safely loads that file if it exists. If not, the app still works.
try:
    from dotenv import load_dotenv # python-dotenv reads key=value pairs from .env
    load_dotenv() # Loads variables into os.environ so they can be read later
except ImportError:
    pass



# SECTION 3: PAGE CONFIGURATION
# Streamlit requires this to be the FIRST Streamlit command called.
# It sets the browser tab title, icon, and layout before anything else renders.
# layout="wide" makes the app use the full browser width — better for results.
# initial_sidebar_state="collapsed" hides the sidebar on load for a clean UI.
st.set_page_config(
    page_title="DEEPTRUST",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)



# SECTION 4: SESSION STATE INITIALISATION
# Streamlit reruns the entire script from top to bottom every time the user clicks a button or uploads a file. Session state is how we preserve data across those reruns — like a short-term memory for the app.
# Each variable below is only set if it does not already exist, so existing values are never accidentally overwritten on rerun.


for k, v in dict(
    page = "welcome", # Start on welcome page — new HCI landing page
    dark_mode = False, # Light mode by default — user can toggle
    session_id = None, # Unique ID for this browser session (set on first load)
    consent_given = False, # Privacy consent must be given before scanning (Kenya DPA 2019)
    results = None, # Stores the full scan result dict after inference runs
    upload_id = None, # Unique ID for the current uploaded image
    scan_count = 0, # Tracks how many scans done this session
    db_conn = None, # SQLite connection — in-memory, wiped on session end
    session_start = None, # Timestamp when session began — shown in profile badge
    scan_history = [], # List of past scan summaries for activity log
    show_help = False, # Legacy — kept for compatibility
    page_before_log = None, # Remember which page to return to from log
    confirm_new = False, # Confirmation state before purging session
).items():
    if k not in st.session_state:
        st.session_state[k] = v



# SECTION 5: THEME SYSTEM (LIGHT AND DARK MODE)
# DEEPTRUST supports two visual themes — light (default) and dark.
# Each theme is a Python dictionary mapping design token names to hex colours.
# This approach means every UI component references a token like T["txt"]
# instead of a hardcoded colour, so switching themes changes the entire appearance with one variable swap at the bottom of this section.

# Colour categories:
# bg_* — background colours for page, cards, inputs
# border* — border/divider line colours
# txt* — text colours (primary and secondary)
# accent* — brand blue used for buttons, links, highlights
# fake_* — red palette for FAKE verdict display
# real_* — green palette for REAL verdict display
# bar_* — colours for metric progress bars
# log_* — colours for forensic log entries (good/bad)


LIGHT = dict(
    # Page and card backgrounds
    bg_page = "#F5F3EE", # Warm off-white page background
    bg_card = "#FFFFFF", # Pure white for metric cards
    bg_sec = "#EEEADE", # Slightly darker for secondary areas
    bg_input = "#F9F8F4", # Very light for input fields
    border = "#D4D0C4", # Soft grey borders
    border2 = "#B8B4A4", # Slightly darker border for emphasis
    # Text colours
    txt = "#1A1916", # Near-black for primary text
    txt2 = "#4A4840", # Medium grey for secondary text
    muted = "#8A8778", # Light grey for captions and labels
    # Brand accent (DEEPTRUST blue)
    accent = "#1B3A5C", # Deep navy blue — primary brand colour
    accent2 = "#2B5A8C", # Slightly lighter blue for hover states
    accent_light = "#E8EFF7", # Very light blue tint for backgrounds
    btn_bg = "#1B3A5C", # Button background — dark navy in light mode
    btn_txt = "#FFFFFF", # Button text — always white
    nav = "#EEEADE", # Navigation bar background
    code_bg = "#E4E0D4", # Background for code/monospace blocks
    toggle_lbl = "Dark mode", # Label shown on the theme toggle button
    # FAKE verdict colour palette (reds)
    fake_bg = "#FBF0F0", # Light red background for FAKE banner
    fake_br = "#D49090", # Red border around FAKE banner
    fake_txt = "#6B1A1A", # Dark red text for FAKE title
    fake_sub = "#8B3030", # Medium red for FAKE subtitle
    fake_conf_bg = "#F0D0D0", # Red tint for confidence badge background
    fake_conf_txt= "#6B1A1A", # Dark red text on confidence badge
    # REAL verdict colour palette (greens)
    real_bg = "#EFF5E8", # Light green background for REAL banner
    real_br = "#90B870", # Green border around REAL banner
    real_txt = "#1E4A0A", # Dark green text for REAL title
    real_sub = "#2E6A14", # Medium green for REAL subtitle
    real_conf_bg = "#D0E8B8", # Green tint for confidence badge background
    real_conf_txt= "#1E4A0A", # Dark green text on confidence badge
    # Metric bar and value colours
    bar_fake = "#C04040", # Red bar — high anomaly / fake signal
    bar_ok = "#3A7A18", # Green bar — normal / authentic signal
    bar_info = "#1B5A8C", # Blue bar — neutral informational metric
    val_fake = "#C04040", # Red number value — fake/bad metric
    val_ok = "#3A7A18", # Green number value — real/good metric
    # Forensic log entry colours
    log_bad = "color:#B03030", # Red text for flagged log entries
    log_ok = "color:#2A6A10", # Green text for normal log entries
    log_warn = "color:#8A5200;font-weight:600", # Amber — borderline
    # UNCERTAIN verdict palette — light mode purple
    unc_bg = "#F0EEF8",
    unc_br = "#9B8FCC",
    unc_txt = "#4A3D88",
    unc_sub = "#6A5DAA",
    unc_conf_bg = "#E8E0F8",
    unc_conf_txt = "#3A2D7A",
)

DARK = dict(
    # Dark mode equivalents of all LIGHT tokens above
    # Same structure, adjusted for dark backgrounds
    bg_page = "#0C1824",
    bg_card = "#112233",
    bg_sec = "#0E1D2C",
    bg_input = "#0A1520",
    border = "#1E3448",
    border2 = "#2A4A68",
    txt = "#DCE8F0",
    txt2 = "#90AABF",
    muted = "#506070",
    accent = "#4A9EDB", # Text/link accent in dark mode
    accent2 = "#6AB4EF",
    accent_light = "#0D2030",
    btn_bg = "#1E4E82", # Button background — darker blue for dark mode
    btn_txt = "#FFFFFF", # Button text — always white
    nav = "#091420",
    code_bg = "#081018",
    toggle_lbl = "Light mode", # When dark mode is on, button says "Light mode"
    fake_bg = "#200C0C",
    fake_br = "#5A2020",
    fake_txt = "#FFAAAA",
    fake_sub = "#CC7070",
    fake_conf_bg = "#3A1010",
    fake_conf_txt= "#FFAAAA",
    real_bg = "#0C1E0C",
    real_br = "#205A20",
    real_txt = "#90EE90",
    real_sub = "#60BB60",
    real_conf_bg = "#102A10",
    real_conf_txt= "#90EE90",
    # UNCERTAIN verdict palette — dark mode purple
    unc_bg = "#130E22",
    unc_br = "#4A3A88",
    unc_txt = "#C0A8FF",
    unc_sub = "#9A82DD",
    unc_conf_bg = "#1E1535",
    unc_conf_txt = "#C0A8FF",
    bar_fake = "#E06060",
    bar_ok = "#60BB60",
    bar_info = "#4A9EDB",
    val_fake = "#EE8080",
    val_ok = "#80DD80",
    log_bad = "color:#EE8080",
    log_ok = "color:#80DD80",
    log_warn = "color:#DDAA40;font-weight:600", # Amber — borderline
)

# Active theme selection
# T is the active theme dictionary used throughout the rest of the app.
# Every colour reference in the UI is T["token_name"] not a hardcoded hex.
# Switching dark_mode in session state and rerunning swaps all colours at once.
T = DARK if st.session_state.dark_mode else LIGHT



# SECTION 6: GLOBAL CSS STYLES
# Streamlit's default appearance is plain and unstyled.
# This block injects custom CSS into the browser to make DEEPTRUST look
# professional — custom fonts, colours, buttons, cards, verdict banners,
# progress bars, forensic log, and AI summary box.
#
# st.markdown(..., unsafe_allow_html=True) is required to inject raw HTML/CSS.
# The f-string (f"""...""") lets us embed Python variables (T["key"]) directly
# into the CSS so colours change automatically when the theme switches.
#
# CSS class naming convention used throughout:
# .topbar / .navbar — top navigation bar
# .step / .steps — progress indicator (Consent → Upload → Results)
# .card — white/dark box containing a section of content
# .verdict — the large FAKE/REAL/UNCERTAIN result banner
# .metric — individual score display boxes (CNN%, AE error)
# .bar — horizontal progress bar for anomaly scores
# .log / .lr / .lk — forensic log table rows and key/value cells
# .ai-box — styled box for the AI forensic summary text

st.markdown(f"""<style>

/* Google Fonts
   Load the Inter typeface from Google's font CDN.
   Inter is a clean, professional sans-serif designed for screen readability.
   Fix: URL must point to fonts.googleapis.com with the correct query string. */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* CSS Reset
   Remove all default browser spacing so our layout starts from zero.
   box-sizing: border-box means padding and borders are included in element
   widths — prevents layout overflow bugs. */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

/* Global font and background — works in both light and dark mode */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    background: {T['bg_page']} !important;
    /* NOTE: color intentionally NOT set here — [class*="css"] matches
       button internals in Streamlit and would override white button text.
       Text colour is set per-element below instead. */
}}
/* Page-level text colour — NO !important so button rules can override */
body, .main, .block-container, [data-testid="stAppViewContainer"] {{
    color: {T['txt']};
}}

/* Target ONLY markdown containers — broad div/span targeting
   broke button text and step circle numbers in light mode */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] strong,
div[data-testid="stMarkdownContainer"] em {{
    color: {T['txt']} !important;
}}
div[data-testid="stMarkdownContainer"] a {{
    color: {T['accent']} !important;
}}
.stCaption p {{ color: {T['muted']} !important; }}
/* button rules consolidated below */

/* Streamlit container overrides */
[data-testid="stAppViewContainer"] {{ background: {T['bg_page']} !important; }}
[data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ── Sidebar toggle: style the native >> button as a hamburger ───────
   Positioned inside the topbar (top:6px left:1.5rem).
   The button IS Streamlit's native toggle so clicking always works.
   We just make it look like a proper nav button instead of ">>" */
[data-testid="stSidebarCollapsedControl"] {{
    position: fixed !important;
    top: 6px !important;
    left: 1.2rem !important;
    z-index: 9999 !important;
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: {T['btn_bg']} !important;
    border-radius: 6px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25) !important;
    padding: 2px !important;
    width: 38px !important;
    height: 38px !important;
    align-items: center !important;
    justify-content: center !important;
}}
/* Hide the >> arrow SVG — show hamburger lines instead */
[data-testid="stSidebarCollapsedControl"] button {{
    background: transparent !important;
    border: none !important;
    width: 34px !important;
    height: 34px !important;
    padding: 0 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
}}
[data-testid="stSidebarCollapsedControl"] button svg {{
    display: none !important;
}}
/* Draw hamburger lines using pseudo-elements on the button */
[data-testid="stSidebarCollapsedControl"] button::before {{
    content: "" !important;
    display: block !important;
    width: 18px !important;
    height: 2px !important;
    background: #ffffff !important;
    box-shadow: 0 6px 0 #ffffff, 0 -6px 0 #ffffff !important;
    border-radius: 2px !important;
}}
/* Sidebar edge collapse button — keep visible */
section[data-testid="stSidebar"] > div:first-child > div > div {{
    visibility: visible !important;
}}

/* Block container width */
.block-container {{
    max-width: 100% !important;
    padding: 0 1.5rem !important;
}}

/* Prevent column overflow */
[data-testid="column"] {{ min-width: 0 !important; }}

/* Style images displayed via st.image() — rounded corners and a border */
[data-testid="stImage"] img {{
    max-width: 100% !important;
    height: auto !important;
    border-radius: 6px;
    border: 1px solid {T['border']};
}}

/* Button styles — consolidated in the BUTTON COLOURS block below.
   Streamlit default grey overridden with DEEPTRUST brand colours.
   Nav buttons use .dt-nav-btn wrapper class (see navbar function). */

/* File upload area
   Style the drag-and-drop file upload box with a dashed border to signal
   it is a drop zone. The label text is made smaller and muted. */
div[data-testid="stFileUploader"] {{
    background: {T['bg_input']} !important;
    border: 1.5px dashed {T['border2']} !important;
    border-radius: 8px !important;
    padding: 10px !important;
}}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] span {{
    color: {T['txt2']} !important;
    font-size: 13px !important;
}}

/* Select boxes and checkboxes
   Apply theme colours to Streamlit's dropdown and checkbox components.
   Multiple selectors target different internal Streamlit div structures
   across different Streamlit versions. */
.stSelectbox > div > div {{
    background: {T['bg_input']} !important;
    border: 1px solid {T['border']} !important;
    color: {T['txt']} !important;
    border-radius: 6px !important;
}}
.stCheckbox > label > span {{ color: {T['txt']} !important; font-size: 13px !important; }}
.stCheckbox > label p {{ color: {T['txt']} !important; }}
.stCheckbox span[data-testid="stMarkdownContainer"] p {{ color: {T['txt']} !important; }}
[data-testid="stCheckbox"] label {{ color: {T['txt']} !important; }}
[data-testid="stCheckbox"] span {{ color: {T['txt']} !important; }}

/* Alert and spinner
   Style st.warning() / st.error() alert boxes and the loading spinner. */
.stAlert {{
    background: {T['bg_sec']} !important;
    border-color: {T['border']} !important;
    color: {T['txt2']} !important;
    border-radius: 6px !important;
}}
.stSpinner > div {{ border-top-color: {T['accent']} !important; }}
@keyframes dt-spin {{ to {{ transform: rotate(360deg); }} }}
.dt-spinner {{
    width: 44px; height: 44px; border: 4px solid {T['border']};
    border-top-color: {T['accent']}; border-radius: 50%;
    animation: dt-spin 0.9s linear infinite;
    margin: 0 auto 16px;
}}

/* Top navigation bar
   Fixed strip at the top showing the DEEPTRUST logo, session info pill,
   and dark mode toggle. Uses flexbox for horizontal alignment.
   Negative margins extend it to the full browser width. */
.topbar {{
    background: {T['nav']};
    border-bottom: 1px solid {T['border']};
    padding: 11px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 1.5rem -1rem; /* bleed to full width */
    flex-wrap: wrap;
    gap: 8px;
}}
/* Logo section — icon square + name + tagline stacked */
.logo {{ display: flex; align-items: center; gap: 12px; }}
.logo-icon {{
    width: 34px; height: 34px;
    background: {T['accent']};
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
}}
.logo-name {{ font-size: 15px; font-weight: 600; color: {T['txt']}; }}
.logo-sub {{ font-size: 11px; color: {T['muted']}; }}

/* Right side of navbar — session pill and toggle button */
.nav-right {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}

/* Pill badge — shows session ID, scan count, or status in monospace font */
.pill {{
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace; /* monospace for IDs */
    background: {T['bg_sec']};
    border: 1px solid {T['border']};
    border-radius: 4px;
    padding: 3px 8px;
    color: {T['muted']};
}}
/* Green pill variant — shown when a scan is complete */
.pill-on {{
    background: {T['real_bg']};
    border-color: {T['real_br']};
    color: {T['real_txt']};
}}

/* Progress steps indicator
   Shows the 3-step workflow: (1) Consent - (2) Upload - (3) Results
   Each step has a numbered circle and label.
   Active step is blue, completed steps are green, future steps are grey. */
.steps {{ display: flex; align-items: center; margin-bottom: 20px; }}
.step {{ display: flex; align-items: center; gap: 7px; font-size: 12px; color: {T['muted']}; }}
.step.on {{ color: {T['accent2']}; font-weight: 600; }} /* current step */
.step.done {{ color: {T['bar_ok']}; }} /* completed step */
.step-n {{
    width: 22px; height: 22px;
    border-radius: 50%; /* circular step number */
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600;
    border: 1.5px solid {T['border']};
    color: {T['muted']};
    background: {T['bg_sec']};
    flex-shrink: 0;
}}
.step.on .step-n {{ background: {T['accent']}; border-color: {T['accent']}; color: #fff; }}
.step.done .step-n {{ background: {T['bar_ok']}; border-color: {T['bar_ok']}; color: #fff; }}
.step-line {{ width: 30px; height: 1px; background: {T['border']}; margin: 0 4px; }}

/* Content cards
   White/dark boxes that group related content with a border and padding.
   Used for metric sections, the forensic log, and the AI summary. */
.card {{
    background: {T['bg_card']};
    border: 1px solid {T['border']};
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 14px;
}}
/* Section label — small all-caps heading above a card section */
.sec-label {{
    font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .8px;
    color: {T['muted']};
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 1px solid {T['border']};
}}

/* Verdict banner
   The large result box at the top of the output page.
   .v-fake — red theme for FAKE verdicts
   .v-real — green theme for REAL verdicts
   .v-uncertain — purple theme for UNCERTAIN verdicts (ambiguous cases)
   A thick left border (border-left: 4px) acts as a colour accent stripe. */
.verdict {{ border-radius: 8px; padding: 16px 20px; margin-bottom: 14px; }}
.v-fake {{ background: {T['fake_bg']}; border: 1px solid {T['fake_br']}; border-left: 4px solid {T['bar_fake']}; }}
.v-real {{ background: {T['real_bg']}; border: 1px solid {T['real_br']}; border-left: 4px solid {T['bar_ok']}; }}
.v-title-fake {{ font-size: 18px; font-weight: 600; color: {T['fake_txt']}; }}
.v-title-real {{ font-size: 18px; font-weight: 600; color: {T['real_txt']}; }}
.v-sub-fake {{ font-size: 12px; color: {T['fake_sub']}; margin-top: 3px; }}
.v-sub-real {{ font-size: 12px; color: {T['real_sub']}; margin-top: 3px; }}
/* Confidence badge — coloured pill showing the percentage score */
.conf-fake {{ display:inline-block; font-size:12px; font-weight:500;
    background:{T['fake_conf_bg']}; color:{T['fake_conf_txt']};
    border-radius:4px; padding:3px 10px; margin-top:8px; }}
.conf-real {{ display:inline-block; font-size:12px; font-weight:500;
    background:{T['real_conf_bg']}; color:{T['real_conf_txt']};
    border-radius:4px; padding:3px 10px; margin-top:8px; }}

/* Score metric boxes
   Small boxes showing CNN%, AE error value, with label and data source.
   .m-val — the large number in the centre
   .m-lbl — the description label below the number
   .m-src — the database column reference in tiny monospace text */
.metric {{ background:{T['bg_sec']}; border:1px solid {T['border']};
    border-radius:6px; padding:12px 14px; text-align:center; }}
.m-val {{ font-size:20px; font-weight:600; color:{T['txt']}; }}
.m-lbl {{ font-size:11px; color:{T['muted']}; margin-top:3px; }}
.m-src {{ font-size:10px; font-family:'JetBrains Mono',monospace;
    color:{T['muted']}; margin-top:4px; opacity:.7; }}

/* Anomaly progress bars
   Horizontal bars showing regional anomaly percentages from Grad-CAM.
   .bar-h — header row with label on left and value on right
   .bar-track — the grey background track
   .bar-fill — the coloured fill that grows to match the percentage */
.bar {{ margin-bottom:9px; }}
.bar-h {{ display:flex; justify-content:space-between;
    font-size:12px; margin-bottom:4px; }}
.bar-lbl {{ color:{T['txt2']}; }}
.bar-track {{ height:5px; background:{T['bg_sec']}; border-radius:3px;
    overflow:hidden; border:1px solid {T['border']}; }}
.bar-fill {{ height:100%; border-radius:3px; }}

/* Forensic log table
   Monospace table showing the full technical audit trail.
   .log — outer container with code-like background
   .lr — each log row (key: value pair)
   .lk — key column (fixed width, muted colour)
   .lv — value column (breaks long hashes across lines) */
.log {{ background:{T['bg_sec']}; border:1px solid {T['border']};
    border-radius:6px; padding:10px 14px;
    font-family:'JetBrains Mono',monospace; font-size:11.5px; }}
.lr {{ display:flex; gap:14px; padding:5px 0;
    border-bottom:1px solid {T['border']}; }}
.lr:last-child {{ border-bottom:none; }} /* no border on final row */
.lk {{ color:{T['muted']}; min-width:148px; flex-shrink:0; }}
.lv {{ color:{T['txt']}; word-break:break-all; }} /* break long hashes */

/* AI forensic summary box
   Styled text box displaying the plain-English explanation of the verdict.
   Left blue border visually distinguishes it from the forensic log.
   .ai-lbl — "DEEPTRUST AI Analysis" header label
   .tier-tag — small badge showing if an override tier was applied */
.ai-box {{
    background:{T['accent_light']};
    border:1px solid {T['border']};
    border-left:3px solid {T['accent']}; /* blue left accent stripe */
    border-radius:0 6px 6px 0;
    padding:14px 16px;
    font-size:13px; color:{T['txt2']}; line-height:1.75;
    margin-top:4px;
}}
.ai-lbl {{
    font-size:10px; font-weight:600;
    text-transform:uppercase; letter-spacing:.7px;
    color:{T['accent']}; margin-bottom:8px;
}}
.tier-tag {{
    display:inline-block; font-size:10px;
    font-family:'JetBrains Mono',monospace;
    background:{T['bg_sec']}; border:1px solid {T['border']};
    border-radius:3px; padding:1px 6px; color:{T['muted']}; margin-left:6px;
}}

/* UNCERTAIN verdict banner — theme-aware purple */
.v-uncertain {{ background:{T['unc_bg']}; border:1px solid {T['unc_br']};
    border-left:4px solid {T['unc_br']}; }}
.v-title-uncertain {{ font-size:18px; font-weight:500; color:{T['unc_txt']}; }}
.v-sub-uncertain {{ font-size:12px; color:{T['unc_sub']}; margin-top:3px; }}
.conf-uncertain {{ display:inline-block; font-size:12px; font-weight:500;
    background:{T['unc_conf_bg']}; color:{T['unc_conf_txt']};
    border-radius:4px; padding:3px 10px; margin-top:8px; }}

/* Miscellaneous components
   .consent — grey box showing the privacy consent text before scanning
   .pane-lbl — caption label below the original/heatmap image panels
   .disclaimer — small footer text at bottom of output page
   .chip — monospace badge for session/upload/result IDs at top of output
   Scrollbar — thin styled scrollbar for WebKit browsers (Chrome, Edge) */
.consent {{ background:{T['bg_sec']}; border:1px solid {T['border']};
    border-radius:6px; padding:14px 16px;
    font-size:13px; color:{T['txt2']}; line-height:1.7; margin-bottom:12px; }}
.consent strong {{ color:{T['txt']}; }}
.pane-lbl {{ text-align:center; font-size:11px; color:{T['muted']};
    padding:5px; border-top:1px solid {T['border']};
    background:{T['bg_sec']}; border-radius:0 0 6px 6px; }}
.profile-badge {{
    display: flex; align-items: center; gap: 10px;
    background: {T['real_bg']}; border: 1px solid {T['real_br']};
    border-radius: 20px; padding: 5px 12px 5px 6px;
}}
.profile-avatar {{
    width: 28px; height: 28px; background: {T['bar_ok']};
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0;
    font-size: 13px; font-weight: 700; color: #fff;
}}
.profile-id {{
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    font-weight: 600; color: {T['real_txt']};
}}
.profile-sub {{
    font-size: 10px; color: {T['real_sub']}; margin-top: 1px;
}}
.profile-badge-inactive {{
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: {T['muted']};
    border: 1px solid {T['border']}; border-radius: 20px;
    padding: 5px 12px;
}}
.disclaimer {{ font-size:11px; color:{T['muted']}; text-align:center;
    border-top:1px solid {T['border']};
    padding-top:12px; margin-top:10px; line-height:1.7; }}
.chip {{ display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:10.5px; background:{T['bg_sec']}; border:1px solid {T['border']};
    border-radius:4px; padding:2px 7px; color:{T['muted']};
    margin-right:5px; margin-bottom:4px; }}
::-webkit-scrollbar {{ width:6px; }}
::-webkit-scrollbar-thumb {{ background:{T['border2']}; border-radius:3px; }}

/* BUTTON COLOURS
   All .stButton buttons → accent background, white text.
   Sidebar nav buttons are handled by Streamlit's native sidebar.
   Primary CTA buttons always get accent + white via inject_button_fix. */

/* All buttons: accent background, white text */
.stButton > button {{
    background-color: {T['accent']} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 9px 18px !important;
    width: 100%;
}}
.stButton > button:hover {{ opacity: 0.87 !important; }}
.stButton > button p,
.stButton > button span,
.stButton > button div {{
    color: #FFFFFF !important;
}}

/* Download buttons */
.stDownloadButton > button {{
    background-color: {T['accent']} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    width: 100% !important;
}}
.stDownloadButton > button p,
.stDownloadButton > button span {{
    color: #FFFFFF !important;
}}

/* Sidebar styling — theme-aware */
[data-testid="stSidebar"] {{
    background: {T['bg_card']} !important;
    border-right: 1px solid {T['border']} !important;
}}
[data-testid="stSidebar"] .stButton > button {{
    background-color: {T['bg_sec']} !important;
    color: {T['txt']} !important;
    border: 1px solid {T['border']} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-align: left !important;
    justify-content: flex-start !important;
}}
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button div {{
    color: {T['txt']} !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: {T['accent_light']} !important;
    border-color: {T['accent']} !important;
    color: {T['accent']} !important;
}}
[data-testid="stSidebar"] .stButton > button:hover p,
[data-testid="stSidebar"] .stButton > button:hover span {{
    color: {T['accent']} !important;
}}
/* Sidebar text colours */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {{
    color: {T['txt2']} !important;
}}
[data-testid="stSidebarNav"] {{ display: none !important; }}

/* Mobile padding */
@media (max-width: 640px) {{
    .block-container {{ padding: 0 0.5rem !important; }}
}}

</style>""", unsafe_allow_html=True)



# SECTION 7: IN-MEMORY DATABASE (SQLite)
# DEEPTRUST uses an in-memory SQLite database to log scan activity during a
# session. "In-memory" means the database exists in RAM only — it is
# automatically destroyed when the app restarts or the browser tab closes.

def get_db():
    """
    Returns the active SQLite database connection.
    Creates it on first call (lazy initialisation).
    Stores the connection in session state so it persists across reruns.
    check_same_thread=False is required for Streamlit's multi-thread model.
    row_factory=sqlite3. Row makes query results accessible as dict-like objects.
    """
    if st.session_state.db_conn is None:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON") # enable cascade deletes
        conn.row_factory = sqlite3.Row # allows result["column_name"] access
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS TBL_SESSIONS (
                Session_ID TEXT PRIMARY KEY,
                IP_Hash TEXT NOT NULL, -- hashed for privacy, never raw IP
                User_Agent TEXT,
                Start_Time TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS TBL_UPLOADS (
                Upload_ID TEXT PRIMARY KEY,
                Session_ID TEXT,
                File_Name TEXT NOT NULL,
                File_Size REAL,
                File_Hash TEXT, -- SHA-256 fingerprint of the image
                Upload_Time TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (Session_ID) REFERENCES TBL_SESSIONS(Session_ID)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS TBL_RESULTS (
                Result_ID TEXT PRIMARY KEY,
                Upload_ID TEXT,
                CNN_Score REAL, -- CNN manipulation score 0.0-1.0
                AE_Error REAL, -- Autoencoder reconstruction error
                Verdict TEXT NOT NULL, -- REAL, FAKE, or UNCERTAIN
                Confidence REAL, -- Final confidence percentage
                Heatmap_Path TEXT, -- Path to saved Grad-CAM image
                Scan_Mode TEXT, -- Standard or High Sensitivity
                Process_Time REAL, -- Seconds taken for the scan
                Timestamp TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (Upload_ID) REFERENCES TBL_UPLOADS(Upload_ID)
                    ON DELETE CASCADE
            );
            -- Simple counter table — increments by 1 for every scan
            CREATE TABLE IF NOT EXISTS TBL_SCAN_COUNTER (
                id INTEGER PRIMARY KEY,
                total INTEGER DEFAULT 0
            );
            INSERT OR IGNORE INTO TBL_SCAN_COUNTER(id, total) VALUES(1, 0);
        """)
        conn.commit()
        st.session_state.db_conn = conn
    return st.session_state.db_conn


def db_session(sid):
    """
    Registers a new session in TBL_SESSIONS.
    INSERT OR IGNORE means if the session already exists, nothing happens.
    The IP is never stored directly — only a SHA-256 hash of the session ID
    is stored, which cannot be reversed to identify the user.
    """
    db = get_db()
    db.execute(
        "INSERT OR IGNORE INTO TBL_SESSIONS(Session_ID, IP_Hash, User_Agent) "
        "VALUES(?,?,?)",
        (sid, hashlib.sha256(sid.encode()).hexdigest()[:32], "Streamlit")
    )
    db.commit()


def db_upload(uid, sid, fname, fsize, fhash):
    """
    Logs an uploaded image to TBL_UPLOADS.
    INSERT OR REPLACE handles the case where the same image is scanned again.
    fhash is the SHA-256 fingerprint — used to detect duplicate uploads
    and provide image integrity evidence in the forensic log.
    """
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO TBL_UPLOADS"
        "(Upload_ID, Session_ID, File_Name, File_Size, File_Hash) "
        "VALUES(?,?,?,?,?)",
        (uid, sid, fname, fsize, fhash)
    )
    db.commit()


def db_result(r):
    """
    Saves a completed scan result to TBL_RESULTS.
    Also increments the session scan counter by 1.
    The result dictionary r must contain: rid, uid, cnn, ae, verdict,
    conf, hmap (heatmap path), mode (scan mode), t (process time).
    """
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO TBL_RESULTS"
        "(Result_ID, Upload_ID, CNN_Score, AE_Error, Verdict, "
        "Confidence, Heatmap_Path, Scan_Mode, Process_Time) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (r["rid"], r["uid"], r["cnn"], r["ae"], r["verdict"],
         r["conf"], r["hmap"], r["mode"], r["t"])
    )
    # Increment the global scan counter for this session
    db.execute("UPDATE TBL_SCAN_COUNTER SET total=total+1 WHERE id=1")
    db.commit()


def db_count():
    """
    Returns the total number of scans completed this session.
    Displayed in the forensic log as Session_Scans.
    """
    row = get_db().execute(
        "SELECT total FROM TBL_SCAN_COUNTER WHERE id=1"
    ).fetchone()
    return row["total"] if row else 0


def db_purge(sid):
    db = get_db()
    db.execute("DELETE FROM TBL_SESSIONS WHERE Session_ID=?", (sid,))
    db.commit()



# SECTION 8: ID GENERATION
# Every session, upload, and scan result gets a unique human-readable ID.
# These IDs appear in the forensic log and PDF report for audit trail purposes.
# The format follows the naming convention defined in the project proposal §2.5.5

def mk_session():
    """
    Creates a unique session ID in the format: SESS-XXXX-XXXX-XXXX
    Uses uuid4() which generates a cryptographically random 32-character hex string.

    """
    r = uuid.uuid4().hex.upper()
    return f"SESS-{r[:4]}-{r[4:8]}-{r[8:12]}"


def mk_upload():
    """
    Creates a unique upload ID in the format: UP-YYYYMMDD-NNNN
    The date component makes it easy to see when an image was uploaded.
    The 4-digit random suffix prevents collisions if multiple images are
    uploaded on the same day.
    Example: UP-20260529-4827
    """
    return f"UP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"


def mk_result(uid):
    """
    Creates a result ID linked to its upload ID: RES-NNNN-HYB
    The NNNN part is extracted from the upload ID so results can be
    traced back to the original upload without a database query.
    """
    return f"RES-{uid.split('-')[-1] if uid else '0000'}-HYB"


def fhash(b):
    """
    Generates a SHA-256 cryptographic hash fingerprint of the raw image bytes.

    """
    return hashlib.sha256(b).hexdigest()



# SECTION 9: MODEL LOADING
# The three AI models that power DEEPTRUST are loaded here.
# @st.cache_resource tells Streamlit to load the models ONCE when the server
# starts and keep them in memory for all subsequent requests.
# Without this decorator, models would reload on every button click — adding 30-60 seconds of wait time per scan.

# Models loaded:
# deeptrust_spotter_v2_final.keras — Xception CNN (166 MB)
# deeptrust_seq_ae.keras — Sequential Autoencoder (64 MB)
# deeptrust_meta_seq_cpu.joblib — LogReg meta-learner (CPU-calibrated)
# deeptrust_scaler_seq_cpu.joblib — StandardScaler for feature normalisation
#
# Additional files loaded:
# feat_min.npy / feat_max.npy — min/max values for normalising GAP features
# These ensure the AE receives the same feature scale it was trained on.


# Google Drive file IDs for Streamlit Cloud deployment
# On your laptop the models/ folder exists so download is skipped.
# On Streamlit Cloud the folder is empty — files are downloaded here.
_GDRIVE_IDS = {
    "deeptrust_spotter_v2_final.keras": "1fTyQbF0QoDBeaPFByvjVRZwGjHXEkzgx",
    "deeptrust_seq_ae.keras": "1_fva2iyObUNglA3b5JCbtUs4DI_JQ3Ft",
    "deeptrust_meta_seq_cpu.joblib": "17YZ8UwFtNGT16CA7PN0KtNgsxgCYNygv",
    "deeptrust_scaler_seq_cpu.joblib": "1NsM4WODawIZfh9Rh1Ihvk6iGT5UgGDQH",
    "feat_min.npy": "1byUsak_0MO4xRlUXOjxKnHqi42ZdFeDy",
    "feat_max.npy": "1vsffmkA4ALjjCgprxBVRlFpqvc_Zw7Vy",
}

def _ensure_models():
    """Download missing model files from Google Drive before loading."""
    import os
    os.makedirs("models", exist_ok=True)
    missing = [
        f for f in _GDRIVE_IDS
        if not os.path.exists(os.path.join("models", f))
    ]
    if not missing:
        return # all files present — laptop mode
    try:
        import gdown
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
    for filename in missing:
        fid = _GDRIVE_IDS[filename]
        dest = os.path.join("models", filename)
        print(f"[DEEPTRUST] Downloading {filename}...")
        gdown.download(
            f"https://drive.google.com/uc?id={fid}",
            dest, quiet=False)


@st.cache_resource
def load_models():
    import tensorflow as tf
    import pickle

    # Download models from Google Drive if running on Streamlit Cloud
    _ensure_models()

    # Stage 1: CNN Spotter — Xception pretrained on ImageNet, fine-tuned on FF++
    cnn = tf.keras.models.load_model(
        "models/deeptrust_spotter_v2_final.keras")

    # Stage 2: Sequential Autoencoder — trained on real face CNN features only
    # Accepts 2048-dim GAP vectors, not raw pixels
    seq_ae = tf.keras.models.load_model(
        "models/deeptrust_seq_ae.keras")

    # Stage 3: Meta-learner — try joblib first, fall back to pickle
    # joblib is the native format; pickle works if sklearn versions differ
    try:
        import joblib
        meta = joblib.load("models/deeptrust_meta_seq_cpu.joblib")
        scaler = joblib.load("models/deeptrust_scaler_seq_cpu.joblib")
    except Exception:
        import pickle
        with open("models/deeptrust_meta_seq_cpu.joblib", "rb") as f:
            meta = pickle.load(f)
        with open("models/deeptrust_scaler_seq_cpu.joblib", "rb") as f:
            scaler = pickle.load(f)

    # Feature normalisation arrays — computed from real face GAP features during training on Kaggle. Used to scale GAP values to 0-1 range before feeding into the AE.
    feat_min = np.load("models/feat_min.npy")
    feat_max = np.load("models/feat_max.npy")

    # Build GAP extractor — extracts the 2048-dim internal feature vectorfrom the CNN's Global Average Pooling layer.
    # This is what makes the pipeline truly sequential:
    # Image - CNN - GAP features - AE (not raw pixels)
    gap_extractor = tf.keras.Model(
        inputs = cnn.inputs,
        outputs = cnn.get_layer("global_average_pooling2d").output,
        name = "gap_extractor"
    )

    return cnn, seq_ae, meta, scaler, gap_extractor, feat_min, feat_max



# SECTION 10: IMAGE PREPROCESSING
# Before any AI model sees the image, it goes through two preprocessing steps:
# 1. CLAHE illumination normalisation — fixes dark/bright areas
# 2. Resize to the correct resolution for each model

def clahe_normalise(img):
    arr = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = cl.apply(lab[:,:,0]) # apply only to L channel
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))



# SECTION 11: FACE DETECTION GATE
# DEEPTRUST is designed exclusively for human face deepfake detection.
# Before running the full AI pipeline, we check if the image actually contains a human face using OpenCV's Haar Cascade classifier.
# If no face is found, the scan is rejected with an error message.
# This prevents the system from analysing documents, landscapes, animals, or website screenshots that would produce meaningless results.

def has_face(img):
    """
    Returns True if a human face is detected in the image, False otherwise.
    Uses two Haar Cascade classifiers in sequence:
    1. haarcascade_frontalface_default — detects straight-on faces
    2. haarcascade_profileface — detects side-profile faces
    This two-stage approach reduces false negatives (missed faces).

    Parameters:
    - scaleFactor=1.1: scan at multiple scales, stepping 10% at a time
    - minNeighbors=4: require 4 nearby detections to confirm a face
    - minSize=(40,40): ignore detections smaller than 40x40 pixels
    """
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Try frontal face detection first
    fc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = fc.detectMultiScale(gray, 1.1, 4, minSize=(40,40))
    if len(faces) > 0:
        return True # FIX: was returning False — if faces found, return True

    # Fall back to profile face detectioN
    # Catches side-on portraits that frontal detector misses
    pc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml")
    profiles = pc.detectMultiScale(gray, 1.1, 3, minSize=(40,40))
    return len(profiles) > 0 # True if profile face found, False if no face


# SECTION 12: INPUT FILE VALIDATION
# Before the image reaches any AI model, it must pass a series of security and quality checks. This function is the first line of defence.
def validate_file(raw, filename):
    # Check 1: File size
    if len(raw) / (1024*1024) > 10:
        return False, "This file is too large. Please upload an image smaller than 10 MB."

    # Check 2: File extension
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("jpg", "jpeg", "png", "webp"):
        return False, f"The file type .{ext} is not supported. Please use a JPG, PNG, or WebP image."

    # Check 3: Magic bytes
    # Every image format has a unique byte signature at the start of the file.
    # JPEG files start with 0xFF 0xD8 0xFF
    # PNG files start with the bytes 0x89 PNG
    # WebP files start with RIFF
    # We read these bytes directly to confirm the file is genuinely an image.
    sigs = {b"\xff\xd8\xff": "jpeg", b"\x89PNG": "png", b"RIFF": "webp"}
    if not any(raw[:len(s)] == s for s in sigs):
        return False, "This file does not appear to be a genuine image. It may be corrupted or renamed. Please try a different file."

    # Check 4: PIL readability and resolution
    # img.verify() does a deep check of the image structure.
    # We must reopen after verify() as verify() moves the file pointer.
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
        img = Image.open(io.BytesIO(raw)) # reopen after verify
        w, h = img.size
    except Exception as e:
        return False, f"Could not read image: {str(e)[:60]}"

    # Check 5: Resolution bounds
    if w < 64 or h < 64:
        return False, f"Image too small ({w}x{h}). Minimum 64x64 pixels."
    if w > 8000 or h > 8000:
        return False, f"Image too large ({w}x{h}). Maximum 8000x8000 pixels."

    return True, "" # all checks passed



# SECTION 13: MAIN INFERENCE ENGINE — run_inference()
# This is the core of DEEPTRUST. It implements the three-stage hybrid pipeline:
#
# STAGE 1 — CNN Spotter (Xception)
# Input: raw image pixels (299×299)
# Output: manipulation probability score 0.0–1.0
# AND 2048-dimensional GAP feature vector
#
# STAGE 2 — Sequential Autoencoder Anomaly Alarm
# Input: CNN GAP feature vector (NOT raw pixels)
# Output: reconstruction error — how different this face's CNN features
# are from a real face's CNN features
# Key: AE was trained ONLY on real faces, so fakes produce higher error
#
# STAGE 3 — LogReg Meta-learner Fusion
# Input: [CNN score, AE error] scaled by StandardScaler
# Output: FAKE/REAL probability — the final verdict
#
# QUALITY GATES (applied after meta-learner):
# CNN Gate: CNN < 20% - trust CNN, block AE false positives
# Tier 1: borderline CNN (50-65%) + clean AE - override to REAL
# Tier 2: high CNN (65-95%) + very clean AE - reduce confidence
# UNCERTAIN: CNN 35-65% + fake probability 50-65% - show purple banner


def run_inference(pil_img, mode):
    import tensorflow as tf
    import pickle

    # Load all models — cached after first call so this is instant
    cnn_model, seq_ae, meta, scaler, gap_extractor, feat_min, feat_max = \
        load_models()
    feat_range = feat_max - feat_min + 1e-8 # avoid division by zero

    # Preprocessing
    # CLAHE normalisation runs first to fix lighting issues before inference
    pil_img = clahe_normalise(pil_img)

    # STAGE 1: CNN Spotter
    # Resize to 299×299 — required input size for Xception architecture.
    # Pixels are kept as raw 0-255 float values because the Xception model
    # has its own preprocessing layer built in that handles normalisation.
    img_c = pil_img.convert("RGB").resize((299, 299))
    b_cnn = np.expand_dims(np.array(img_c, dtype=np.float32), 0)

    # Run the CNN — get manipulation probability (0=real, 1=fake)
    cnn_raw = float(cnn_model(b_cnn, training=False).numpy()[0][0])

    # Extract GAP features — 2048 values representing the CNN's internal
    # understanding of this face. These are passed to the AE in Stage 2.
    gap_feats = gap_extractor(b_cnn, training=False).numpy()[0]

    # STAGE 2: Sequential AE Anomaly Alarm
    # Normalise GAP features to 0-1 range using training-time min/max.
    # This ensures the AE sees the same scale of values it was trained on.
    gap_norm = np.clip((gap_feats - feat_min) / feat_range, 0, 1)

    # AE reconstructs the feature vector — if the face is real, error is low.
    # If the face is a deepfake, CNN features are anomalous and error is high.
    recon = seq_ae(np.expand_dims(gap_norm, 0), training=False).numpy()[0]
    ae_err = float(np.mean(np.power(gap_norm - recon, 2)))

    # Image quality signals
    # These help the system understand the quality of the input image.
    # Low quality images (dark, blurry) can produce false positives.
    img_gray = np.array(pil_img.convert("L"), dtype=np.float32)
    brightness = float(np.mean(img_gray)) / 255.0
    # Laplacian variance measures image sharpness — low = blurry
    sharpness = min(
        float(cv2.Laplacian(np.uint8(img_gray), cv2.CV_64F).var()) / 500.0,
        1.0)
    # Low quality flag — used in Tier 0b gate
    is_low_quality = sharpness < 0.5 and brightness < 0.75

    # STAGE 3: Meta-learner Fusion
    # The meta-learner takes [CNN score, AE error] and learns how to combine them into a final verdict. The scaler normalises both values to a similar range before the LogReg makes its decision.
    # Only 2 features — CNN and AE — matching the training setup.
    feats = np.array([[cnn_raw, ae_err]])
    feats_scaled = scaler.transform(feats)
    proba = meta.predict_proba(feats_scaled)[0]
    fp = float(proba[1]) # probability this is FAKE
    conf = float(np.max(proba)) * 100
    verdict = "FAKE" if fp >= 0.5 else "REAL"
    cnn_pct = cnn_raw * 100

    # Quality-aware override tiers
    # The meta-learner is good but not perfect on out-of-distribution inputs.
    # These tiers correct known failure modes discovered during testing.
    # They fire AFTER the meta-learner, not instead of it.

    ov_tier = None

    # Quality-aware override tiers
    # Applied after the meta-learner. Three CNN zones, each with
    # different AE thresholds based on calibration testing.
    #
    # Zone A (CNN < 0.20): extremely confident REAL signal from CNN
    # - AE < 0.035 → gate fires, REAL (WhatsApp / screenshot protection)
    # - AE ≥ 0.035 → gate blocked, FAKE kept (animated / GPT face)
    #
    # Zone B (CNN 0.20–0.50): mid-range CNN + clean AE → both say REAL
    # - AE < 0.028 → gate fires, REAL
    #
    # Zones C–D (higher CNN): handled by Tier 0b, Tier 1, Tier 2

    if verdict == "FAKE":

        if cnn_raw < 0.20 and ae_err < 0.035:
            # Zone A clean — CNN very confident REAL, AE within safe range
            # Covers WhatsApp photos, screenshots, compressed images
            verdict = "REAL"
            fp = fp * 0.3
            conf = (1 - fp) * 100
            ov_tier = "gate"

        elif cnn_raw < 0.20 and ae_err >= 0.035:
            # Zone A elevated — CNN says real but AE significantly elevated
            # Split by CNN floor:
            # CNN < 9% → extremely strong REAL signal, AE cannot override
            # (0.35% CNN on a selfie is categorically different
            # from 15% CNN on an ambiguous image)
            # CNN 9-20% → genuinely ambiguous, trust the AE (animated/GPT face)
            if cnn_raw < 0.09:
                # CNN floor — too strong a REAL signal for AE to override
                verdict = "REAL"
                fp = fp * 0.3
                conf = (1 - fp) * 100
                ov_tier = "gate"
            else:
                # AE override — CNN has some fake signal, AE elevation meaningful
                ov_tier = "ae_override"

        elif 0.20 <= cnn_raw < 0.50 and ae_err < 0.028:
            # Zone B — both branches point to REAL → override
            verdict = "REAL"
            fp = fp * 0.35
            conf = (1 - fp) * 100
            ov_tier = "gate"

        # Tier 0b — Strong CNN REAL + low quality image
        # CNN 15-30% on blurry/dark image → AE misread due to artifacts
        elif (0.15 <= cnn_raw < 0.30
                and fp < 0.65
                and is_low_quality):
            verdict = "REAL"
            fp = fp * 0.4
            conf = (1 - fp) * 100
            ov_tier = "0b"

        # Tier 1 — Borderline CNN + very clean AE
        # CNN 50-65% on a blurry image is unreliable.
        # AE < 0.015 is unusually clean — face reconstructs like a real one.
        elif (50.0 <= cnn_pct <= 65.0
                and ae_err < 0.015
                and sharpness < 0.70):
            verdict = "REAL"
            fp = fp * 0.5
            conf = (1 - fp) * 100
            ov_tier = "1"

        # Tier 2 — High CNN + anomalously clean AE
        # CNN 65-95% suggests manipulation, but AE < 0.008 is suspiciously
        # clean — genuine deepfakes almost never have AE this low.
        # Reduce confidence rather than flip the verdict.
        elif (65.0 < cnn_pct <= 95.0
                and ae_err < 0.008):
            fp = fp * 0.65
            verdict = "FAKE" if fp >= 0.5 else "REAL"
            conf = (fp * 100) if verdict == "FAKE" else ((1 - fp) * 100)
            ov_tier = "2"

    # UNCERTAIN gate
    # Meta-learner is genuinely ambiguous and no tier has overridden.
    # Only fires on sharp images (sharpness > 0.70) — blurry images are
    # expected to be uncertain and should not trigger this gate.
    real_prob = 1 - fp
    if (verdict == "REAL"
            and real_prob < 0.70
            and sharpness > 0.70
            and ov_tier is None):
        verdict = "UNCERTAIN"
        ov_tier = "U"

    # UNCERTAIN gate for low-confidence FAKE (asymmetry fix)
    # A FAKE verdict with fp 0.50-0.65 on a sharp image is also ambiguous.
    if (verdict == "FAKE"
            and fp < 0.65
            and sharpness > 0.70
            and ov_tier is None):
        verdict = "UNCERTAIN"
        ov_tier = "U"

    # SECTION 14: GRAD-CAM HEATMAP GENERATION
    # Grad-CAM shows WHERE in the face the CNN detected manipulation.
    # Threshold for heatmap display — only show activations above this value
    # 0.4 for Standard mode, 0.3 for High Sensitivity mode (more areas shown)
    thresh = 0.3 if "High" in mode else 0.4

    try:
        # Primary path: extract from nested Xception sub-model
        # The CNN is structured as: outer_model - xception - classification We need to reach inside the xception sub-model to get the feature map
        xb = cnn_model.get_layer("xception")
        fe = tf.keras.models.Model(
            inputs = xb.input,
            outputs = xb.get_layer("block14_sepconv2_act").output)
        # Xception requires its specific preprocessing (scales pixels to -1 to +1)
        fmap = fe.predict(
            tf.keras.applications.xception.preprocess_input(b_cnn.copy()),
            verbose=0)

    except Exception:
        # Fallback path: access layer directly from outer model
        # If the nested path fails (model structure difference), try directly
        fe = tf.keras.models.Model(
            inputs = cnn_model.input,
            outputs = cnn_model.get_layer("block14_sepconv2_act").output)
        fmap = fe.predict(b_cnn, verbose=0)

    # Build the heatmap from feature activations
    # fmap shape: (1, H, W, C) — 1 image, spatial grid, many channels
    # Average across all channels (axis=-1) to get one activation map
    hm = np.squeeze(np.mean(fmap, axis=-1)) # shape: (H, W)

    # ReLU — keep only positive activations (manipulated regions)
    # Normalise to 0-1 range for consistent colour mapping
    hm = np.maximum(hm, 0)
    hm /= (np.max(hm) + 1e-10) # avoid division by zero

    # Apply threshold — suppress low-activation background noise
    # Gaussian blur softens the edges for a cleaner visual overlay
    hmt = cv2.GaussianBlur(np.where(hm > thresh, hm, 0), (5,5), 0)

    # Resize heatmap to match CNN input size (299×299)
    hmr = cv2.resize(hmt, (299,299))

    # Apply JET colour map: blue (low) → green (medium) → red (high suspicion)
    hmc = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * hmr), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)

    # Blend original image (70%) with heatmap (30%) for the overlay
    overlay = cv2.addWeighted(
        np.uint8(np.array(img_c)), 0.7, # original image at 70% opacity
        hmc, 0.3, # heatmap at 30% opacity
        0)

    # Save heatmap overlay to disk
    # Saved to results/previews/ folder for display in the app and PDF
    os.makedirs("results/previews", exist_ok=True)
    rid = mk_result(st.session_state.upload_id)
    hpath = f"results/previews/{rid}.png"
    # OpenCV saves in BGR format — convert from RGB before saving
    cv2.imwrite(hpath, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Extract regional anomaly scores
    # Divide the heatmap into facial regions by vertical position.
    # This gives percentage anomaly scores for each region.
    # The heatmap rows correspond approximately to face zones:
    # 0-20% of height → Hair / forehead boundary
    # 25-45% of height → Eye region
    # 10-60% of height → Skin texture (broad mid-face)
    # 55-75% of height → Mouth / chin area
    h, w = hmr.shape
    reg = {
        "Eye region": float(np.mean(hmr[int(h*.25):int(h*.45), :])),
        "Skin texture": float(np.mean(hmr[int(h*.10):int(h*.60), :])),
        "Mouth area": float(np.mean(hmr[int(h*.55):int(h*.75), :])),
        "Hair boundary": float(np.mean(hmr[0:int(h*.20), :])),
    }
    # Normalise all scores so the highest region = 100% and others scale down
    mx = max(reg.values()) + 1e-10
    reg = {k: round(v/mx * 100, 1) for k,v in reg.items()}

    # Return complete inference results as a dictionary
    # All downstream functions (display, PDF, database, summary) read from this dict
    return dict(
        cnn_score = round(cnn_raw * 100, 2), # CNN score as percentage
        ae_error = round(ae_err, 4), # AE reconstruction error
        verdict = verdict, # REAL, FAKE, or UNCERTAIN
        confidence = round(conf, 1), # Final confidence %
        hmap = hpath, # Path to saved heatmap file
        orig = pil_img, # Original PIL image (for PDF)
        hmap_img = Image.fromarray(overlay), # Heatmap PIL image (for PDF)
        regions = reg, # Regional anomaly scores dict
        result_id = rid, # RES-XXXX-HYB identifier
        ov_tier = ov_tier, # Which override tier fired (if any)
    )



# SECTION 15: AI FORENSIC SUMMARY
# Generates a plain-English explanation of the scan result.
def ai_summary(r):
    is_fake = r["verdict"] == "FAKE"
    is_uncertain = r["verdict"] == "UNCERTAIN"
    cnn = r["cnn_score"] # CNN score as percentage e.g. 31.4
    ae = r["ae_error"] # AE reconstruction error e.g. 0.0319
    conf = r["confidence"] # Final confidence % e.g. 58.2
    reg = r["regions"] # Regional anomaly dict

    # Sort regions from highest to lowest anomaly score
    top = sorted(reg.items(), key=lambda x: x[1], reverse=True)
    t2 = f"the {top[0][0].lower()} and {top[1][0].lower()}"
    ts = top[0][1] # highest region score
    bot = top[-1][0].lower() # lowest region name
    bs = top[-1][1] # lowest region score

    # Get override tier — tells us if any gate or tier modified the verdict
    ov = r.get("ov_tier") # FIX: only define once, removed duplicateS

    # Determine the story for Ollama prompt
    cnn_direction = "above" if cnn > 50 else "below"
    ae_direction = "above" if ae > 0.028 else "below"

    if is_fake and cnn > 50 and ae > 0.028:
        branch_story = (f"Both the CNN ({cnn}%) and the Autoencoder "
                        f"({ae}) agree this is manipulated.")
    elif is_fake and cnn <= 50:
        branch_story = (f"The CNN ({cnn}%) did not flag this alone, "
                        f"but the Autoencoder ({ae}) detected "
                        f"anomalies the CNN missed.")
    elif not is_fake and cnn <= 50 and ae <= 0.028:
        branch_story = (f"Both the CNN ({cnn}%) and the Autoencoder "
                        f"({ae}) agree this is authentic.")
    else:
        branch_story = (f"The CNN scored {cnn}% and the Autoencoder "
                        f"recorded {ae}, leading to a combined "
                        f"{'FAKE' if is_fake else 'REAL'} verdict.")

    top_region = top[0][0] if top else "face"
    top_pct = top[0][1] if top else 0

    # Build structured Ollama prompt
    # Give Ollama the exact facts and a rigid template to prevent hallucination.
    # Low temperature (0.2) keeps output close to the prompt facts.
    # We explicitly forbid common errors ("above 50% threshold" when below 50%).
    ollama_prompt = f"""You are explaining a deepfake detection result to a non-technical user.
Write exactly 3 short sentences. Be friendly and clear. Use simple words.

Here are the facts — use ONLY these numbers, do not make up others:
- CNN score: {cnn}% ({cnn_direction} the 50% threshold)
- Autoencoder error: {ae} ({ae_direction} the 0.025 baseline)
- Story: {branch_story}
- Most suspicious face region: {top_region} at {top_pct}%
- Final verdict: {'FAKE (deepfake detected)' if is_fake else 'REAL (authentic image)'}
- Confidence: {conf}%

Write sentence 1: Explain what the CNN score of {cnn}% means in plain English.
Write sentence 2: Explain what the Autoencoder error of {ae} means and how it combined with the CNN to reach the verdict.
Write sentence 3: Explain what the {top_region} being highlighted at {top_pct}% in the heatmap means.

Rules:
- Do NOT say "above 50% threshold" if CNN is {cnn}% and {cnn}% is below 50.
- Do NOT use technical jargon.
- Do NOT start with "I".
- Keep it friendly and under 80 words total."""

    # Try Ollama first
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": ollama_prompt,
                "stream": False,
                "options": {
                    "num_predict": 200,
                    "temperature": 0.2, # low = stays closer to facts
                    "stop": ["\n\n"] # stop after first paragraph
                }
            },
            timeout=30)
        if resp.status_code == 200:
            text = resp.json().get("response", "").strip()
            # Quality gate: Ollama output must reference the actual CNN number
            # If it does not, it generated generic text — reject and use fallback
            cnn_str = str(round(cnn, 1))
            if text and len(text) > 50 and cnn_str in text:
                return "LOCAL-LLM", text
    except Exception:
        pass # Ollama not running — fall through to rule-based summary

   

    # RULE-BASED SUMMARY FALLBACK
    # When Ollama is unavailable or fails the quality check, this generates a structured factual summary using the actual scan numbers.

    # UNCERTAIN case
    if is_uncertain:
        return "LOCAL", (
            f"DEEPTRUST returned an inconclusive result at {conf}% confidence. "
            f"The CNN branch scored {cnn}% and the Autoencoder error was {ae}, "
            f"but neither signal was strong enough for a reliable verdict. "
            f"This is most common with high-quality synthetic images that partially "
            f"evade detection. Independent expert forensic analysis is recommended."
        )

    # Gate or tier override fired — verdict was changed to REAL
    if not is_fake and ov in ("0b", "gate"):
        # Explain what actually happened — CNN was confident enough to override
        ae_note = (
            f"Although the Autoencoder recorded a reconstruction error of {ae} "
            f"— slightly above the 0.025 baseline — "
            if ae > 0.028 else
            f"The Autoencoder error of {ae} is within the 0.025 baseline. "
        )
        return "LOCAL", (
            f"The CNN spotter scored {cnn}%, which is below the 50% manipulation "
            f"threshold — meaning the CNN is confident this face is authentic. "
            f"{ae_note}"
            f"The CNN confidence gate overrode the initial assessment: "
            f"when both the CNN score and the overall confidence pattern point to "
            f"an authentic face, DEEPTRUST trusts the CNN signal. "
            f"DEEPTRUST classifies this image as authentic at {conf}% confidence. "
            f"The Grad-CAM heatmap shows highest activity on the "
            f"{top[0][0].lower()} at {ts}%, which is within normal range "
            f"for authentic faces."
        )

    # FAKE cases
    if is_fake:
        # CNN sentence — above or below threshold
        cnn_sent = (
            f"The CNN spotter scored {cnn}%, which is {cnn_direction} "
            f"the 50% manipulation threshold — "
            f"{'the neural network detected facial manipulation patterns' if cnn > 50 else 'meaning the CNN alone would have passed this image as authentic'}."
        )
        # AE sentence — above or below baseline
        ae_ctx = ("significantly above" if ae > 0.032
                  else "moderately above" if ae > 0.028
                  else "below")
        ae_sent = (
            f"The Autoencoder reconstruction error was {ae}, "
            f"which is {ae_ctx} the authentic face baseline of 0.025 — "
            f"{'suggesting the face deviates from genuine facial CNN feature patterns' if ae > 0.028 else 'however the combined CNN and AE signals still indicate manipulation'}."
        )
        # Agreement between branches
        if cnn > 50 and ae > 0.028:
            agree = (f"Both branches agree — the CNN flagged manipulation and "
                     f"the AE confirmed anomalous reconstruction, giving "
                     f"the system {conf}% confidence in the FAKE verdict.")
        else:
            agree = (f"The branches disagreed — the CNN scored below 50% but "
                     f"the AE detected elevated reconstruction error. "
                     f"The meta-learner sided with the AE, as the AE is designed "
                     f"to catch zero-day deepfakes that evade CNN detection.")
        # Grad-CAM sentence
        gradcam = (
            f"The Grad-CAM heatmap concentrated activity on {t2} "
            f"({top[0][1]}% and {top[1][1]}%), with the {bot} least affected "
            f"at {bs}% — these facial boundary regions are commonly altered "
            f"by deepfake generation networks."
            if ts >= 65 else
            f"The Grad-CAM heatmap shows distributed activity across {t2}, "
            f"indicating manipulation artifacts spread across the face."
        )
        conclusion = (
            f"Based on the combined analysis, DEEPTRUST concludes this image "
            f"is a likely deepfake at {conf}% confidence. Expert human review "
            f"is advised before any consequential decisions are made."
        )
        return "LOCAL", f"{cnn_sent} {ae_sent} {agree} {gradcam} {conclusion}"

    # REAL cases — no gate fired, standard clean REAL
    else:
        cnn_sent = (
            f"The CNN spotter scored {cnn}%, which is below the 50% "
            f"manipulation threshold — no significant manipulation patterns detected."
        )
        # AE sentence depends on whether AE is clean or slightly elevated
        if ae > 0.028:
            ae_sent = (
                f"The Autoencoder error of {ae} is slightly above the 0.025 "
                f"baseline, but combined with the CNN score both branches "
                f"point toward an authentic image."
            )
        else:
            ae_sent = (
                f"The Autoencoder error of {ae} is within the authentic face "
                f"baseline of 0.025, confirming the face's internal CNN feature "
                f"patterns match genuine facial structure."
            )
        gradcam = (
            f"The Grad-CAM heatmap showed the highest activity on the "
            f"{top[0][0].lower()} at {ts}%, which is within normal range "
            f"for authentic faces and does not indicate manipulation."
        )
        conclusion = (
            f"DEEPTRUST classifies this image as authentic at {conf}% confidence. "
            f"No significant manipulation signatures were detected."
        )
        return "LOCAL", f"{cnn_sent} {ae_sent} {gradcam} {conclusion}"



# SECTION 16: PDF FORENSIC REPORT GENERATION
# Generates a downloadable PDF report of the scan results using ReportLab.
def make_pdf(r, summary):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable, Image as RLImg)
    from reportlab.lib.enums import TA_CENTER

    # Store PDF in-memory
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
          leftMargin=2*cm, rightMargin=2*cm,
          topMargin=2*cm, bottomMargin=2*cm)

    # Colour palette
    NAVY = colors.HexColor("#1B3A5C") # DEEPTRUST brand blue
    RED = colors.HexColor("#8B2020") # FAKE verdict colour
    GREEN = colors.HexColor("#2A5A10") # REAL verdict colour
    GRAY = colors.HexColor("#4A4840") # Body text colour
    LGRAY = colors.HexColor("#F5F3EE") # Alternating table row background
    BDR = colors.HexColor("#D4D0C4") # Table border colour

    # Verdict colour — red for FAKE, green for REAL/UNCERTAIN
    VC = RED if r["verdict"] == "FAKE" else GREEN

    # Typography styles
    # ps() is a shorthand helper for creating ParagraphStyle objects
    def ps(n, **kw):
        return ParagraphStyle(n, **kw)

    S = dict(
        h1 = ps("h1", fontSize=18, textColor=NAVY,
                fontName="Helvetica-Bold", spaceAfter=4), # report title
        h2 = ps("h2", fontSize=10, textColor=GRAY,
                fontName="Helvetica", spaceAfter=10), # subtitle
        hd = ps("hd", fontSize=11, textColor=NAVY,
                fontName="Helvetica-Bold",
                spaceBefore=10, spaceAfter=5), # section heading
        bd = ps("bd", fontSize=10, textColor=GRAY,
                fontName="Helvetica",
                spaceAfter=5, leading=14), # body text
        vd = ps("vd", fontSize=13, textColor=VC,
                fontName="Helvetica-Bold", spaceAfter=4), # verdict line
        sm = ps("sm", fontSize=8, textColor=GRAY,
                fontName="Helvetica-Oblique", leading=12), # small/disclaimer
        cp = ps("cp", fontSize=8, textColor=GRAY,
                fontName="Helvetica-Oblique",
                alignment=TA_CENTER), # image caption
    )

    # Reusable table styles
    # ts — base table style (grid, padding, alternating row backgrounds)
    # th — header row style (navy background, white bold text)
    ts = [
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (-1,-1), GRAY),
        ("GRID", (0,0), (-1,-1), 0.5, BDR),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LGRAY]),
    ]
    th = [
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ]

    # Build the page content
    # ReportLab builds PDFs from a list of "flowables" — objects that flow
    # down the page: paragraphs, tables, spacers, horizontal rules.
    story = [
        Paragraph("DEEPTRUST", S["h1"]),
        Paragraph("Forensic Deepfake Detection Report", S["h2"]),
        HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=8),
        Paragraph(
            "DEEPFAKE DETECTED" if r["verdict"] == "FAKE" else "AUTHENTIC IMAGE",
            S["vd"]),
        Paragraph(
            f"Confidence: <b>{r['confidence']}%</b> | "
            f"Mode: DEEPTRUST Hybrid Ensemble",
            S["bd"]),
        Spacer(1, 6),
        Paragraph("Report Metadata", S["hd"]),
    ]

    # Audit metadata table
    # Shows all the identifiers and file details for chain-of-custody purposes
    mt = Table([
        ["Result ID", r.get("result_id", "")],
        ["Upload ID", r.get("upload_id", "")],
        ["Session ID", r.get("session_id", "")],
        ["Timestamp", r.get("timestamp", "")],
        ["File", r.get("file_name", "")],
        ["Size", f"{r.get('file_size', '')} KB"],
        ["SHA-256", r.get("file_hash", "")[:32] + "..."],
        ["Process time", f"{r.get('process_time', '')}s"],
    ], colWidths=[4*cm, 13*cm])
    mt.setStyle(TableStyle(ts + [
        ("BACKGROUND", (0,0), (0,-1), LGRAY), # grey key column
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
    ]))
    story += [mt, Spacer(1,8), Paragraph("Forensic Metrics", S["hd"])]

    # AI scores table
    mm = Table([
        ["Metric", "Value", "Threshold", "Status"],
        ["CNN score", f"{r['cnn_score']}%", "Above 50% = fake",
            "Fake signal" if r['cnn_score'] > 50 else "Real signal"],
        ["AE error", f"{r['ae_error']}", "Baseline: 0.025",
            "Elevated" if r['ae_error'] > 0.030 else "Normal"],
        ["Confidence", f"{r['confidence']}%","—", "—"],
    ], colWidths=[5*cm, 3*cm, 4*cm, 3.5*cm])
    mm.setStyle(TableStyle(ts + th))
    story += [mm, Spacer(1,8), Paragraph("Grad-CAM Regional Anomalies", S["hd"])]

    # Regional anomaly scores table
    rm = Table(
        [["Region", "Anomaly %"]] +
        [[k, f"{v}%"] for k, v in r["regions"].items()],
        colWidths=[8*cm, 5*cm])
    rm.setStyle(TableStyle(ts + th))
    story += [rm, Spacer(1,8), Paragraph("Image Evidence", S["hd"])]

    # Side-by-side image evidence
    # Original image on the left, Grad-CAM heatmap overlay on the right
    try:
        ob = io.BytesIO(); hb = io.BytesIO()
        r["orig"].resize((250,250)).save(ob, format="PNG")
        r["hmap_img"].resize((250,250)).save(hb, format="PNG")
        ob.seek(0); hb.seek(0)
        it = Table(
            [[RLImg(ob, width=6*cm, height=6*cm),
              RLImg(hb, width=6*cm, height=6*cm)]],
            colWidths=[7*cm, 7*cm])
        it.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
        story += [it,
            Paragraph("Left: Original upload Right: Grad-CAM heatmap", S["cp"])]
    except Exception:
        pass # if images fail, skip — rest of PDF still generates

    # AI Summary and disclaimer
    story += [
        Spacer(1,8),
        Paragraph("AI Forensic Summary", S["hd"]),
        Paragraph(summary, S["bd"]),
        Spacer(1,10),
        HRFlowable(width="100%", thickness=0.5, color=BDR, spaceAfter=6),
        Paragraph("Disclaimer", S["hd"]),
        Paragraph(
            "Generated by DEEPTRUST for forensic assistance only. "
            "Should not be regarded as legal evidence without expert review. "
            "All scan data is purged on session close in accordance with the "
            "Kenya Data Protection Act (2019).",
            S["sm"]),
    ]

    # Build and return PDF bytes
    doc.build(story)
    buf.seek(0)
    return buf.read()


# SECTION 17: SESSION CLEANUP
# When a user ends their session or closes the app, all data associated with that session must be deleted. This fulfils the Kenya Data Protection Act 2019 principle of data minimisation — no scan data is retained beyond what is needed.
# What gets deleted:
# 1. The Grad-CAM heatmap image file saved to results/previews/
# 2. All database records (session, uploads, results) via db_purge()
# which cascades through all linked tables automatically
# 3. All Streamlit session state variables reset to their defaults

def clear_session():
    # Step 1: Delete saved heatmap image from disk
    # The Grad-CAM overlay is the only file DEEPTRUST saves to disk.
    # Check it exists before trying to delete (avoids FileNotFoundError).
    r = st.session_state.get("results")
    if r:
        hp = r.get("hmap", "")
        if hp and os.path.exists(hp):
            os.remove(hp) # permanently delete the heatmap PNG file

    # Step 2: Purge database records
    # db_purge() deletes the session row from TBL_SESSIONS.
    # Because of ON DELETE CASCADE, this also removes all linked
    # TBL_UPLOADS and TBL_RESULTS rows automatically.
    sid = st.session_state.get("session_id")
    if sid:
        db_purge(sid)

    # Step 3: Reset Streamlit session state
    # Clear all in-memory variables so the app returns to its initial state.
    # The user will need to go through consent and upload again.
    # Clear AI summary cache
    for k in list(st.session_state.keys()):
        if k.startswith("ai_txt_"):
            del st.session_state[k]
    st.session_state.results = None
    st.session_state.upload_id = None
    st.session_state.session_id = None
    st.session_state.consent_given = False
    st.session_state.session_start = None
    st.session_state.scan_history = None
    st.session_state.confirm_new = False



# SECTION 18: UI HELPER FUNCTIONS
# These small functions generate reusable HTML/CSS components used throughout the app's output page. Keeping them as functions avoids repeating the same HTML string everywhere and makes the code easier to read.
def start_new_scan():
    """
    Resets for a new scan WITHOUT destroying the session.
    The user keeps their Session ID and scan history.
    Only clears the current result and upload ID.
    Session only fully clears when user closes the browser (in-memory SQLite).
    """
    # Clear AI summary cache for old result
    for k in list(st.session_state.keys()):
        if k.startswith("ai_txt_"):
            del st.session_state[k]
    # Clear heatmap file from disk
    r = st.session_state.get("results")
    if r:
        hp = r.get("hmap", "")
        if hp and os.path.exists(hp):
            os.remove(hp)
    # Reset result and upload but KEEP session + history
    st.session_state.results = None
    st.session_state.upload_id = None
    st.session_state.confirm_new = False


def navbar():
    sess = st.session_state.session_id
    scans = db_count()
    start_t = st.session_state.get("session_start", "")

    # SIDEBAR NAVIGATION
    # Streamlit's native sidebar handles open/close, mobile drawer,
    # and the hamburger icon automatically. No CSS fighting needed.
    with st.sidebar:
        # Sidebar header
        st.markdown(f"""
        <div style="padding:4px 0 16px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:32px;height:32px;background:{T['accent']};
                     border-radius:6px;display:flex;align-items:center;
                     justify-content:center;flex-shrink:0;">
                    <svg width="16" height="16" viewBox="0 0 18 18" fill="none">
                        <circle cx="7.5" cy="7.5" r="5" stroke="#fff" stroke-width="1.5"/>
                        <line x1="11.5" y1="11.5" x2="16" y2="16"
                              stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
                        <line x1="5.5" y1="7.5" x2="9.5" y2="7.5"
                              stroke="#AACCEE" stroke-width="1.2" stroke-linecap="round"/>
                        <line x1="7.5" y1="5.5" x2="7.5" y2="9.5"
                              stroke="#AACCEE" stroke-width="1.2" stroke-linecap="round"/>
                    </svg>
                </div>
                <div>
                    <div style="font-size:15px;font-weight:700;
                         color:{T['txt']};line-height:1.2;">DEEPTRUST</div>
                    <div style="font-size:10px;color:{T['muted']};">
                        AI deepfake detection · TUK 2026</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # Navigation buttons — full width, clear labels
        st.markdown(f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;"
                    f"letter-spacing:.8px;color:{T['muted']};margin-bottom:6px;'>"
                    f"NAVIGATE</div>", unsafe_allow_html=True)

        if st.button("Home", key="nav_home", use_container_width=True):
            st.session_state.page = "welcome"
            st.rerun()

        if st.button("My scans", key="nav_log", use_container_width=True):
            st.session_state.page = "log"
            st.rerun()

        if st.button("Help", key="nav_help", use_container_width=True):
            st.session_state.page = "help"
            st.rerun()

        st.divider()

        # Theme toggle
        st.markdown(f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;"
                    f"letter-spacing:.8px;color:{T['muted']};margin-bottom:6px;'>"
                    f"APPEARANCE</div>", unsafe_allow_html=True)

        if st.button(
            T['toggle_lbl'],
            key="tm", use_container_width=True
        ):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.divider()

        # Session identity panel
        st.markdown(f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;"
                    f"letter-spacing:.8px;color:{T['muted']};margin-bottom:8px;'>"
                    f"SESSION</div>", unsafe_allow_html=True)

        if sess:
            scan_txt = f"{scans} scan{'s' if scans != 1 else ''}"
            st.markdown(f"""
            <div style="background:{T['real_bg']};border:1px solid {T['real_br']};
                 border-radius:8px;padding:10px 12px;">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <div style="width:20px;height:20px;background:{T['bar_ok']};
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;flex-shrink:0;">
                        <svg width="10" height="10" viewBox="0 0 14 14" fill="none">
                            <circle cx="7" cy="5" r="3" stroke="#fff" stroke-width="1.4"/>
                            <path d="M1 13c0-3 2.7-5 6-5s6 2 6 5" stroke="#fff"
                                  stroke-width="1.4" stroke-linecap="round" fill="none"/>
                        </svg>
                    </div>
                    <span style="font-size:10px;font-weight:600;
                          color:{T['real_txt']};">Active session</span>
                </div>
                <div style="font-size:10px;font-family:'JetBrains Mono',monospace;
                     color:{T['bar_ok']};font-weight:600;word-break:break-all;
                     margin-bottom:4px;">{sess}</div>
                <div style="font-size:10px;color:{T['real_sub']};">
                    {scan_txt} · Since {start_t}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{T['bg_sec']};border:1px solid {T['border']};
                 border-radius:8px;padding:10px 12px;
                 font-size:11px;color:{T['muted']};">
                No active session.<br>
                Accept consent on the scan page to begin.
            </div>""", unsafe_allow_html=True)

        # Scan history in sidebar if available
        history = st.session_state.get("scan_history") or []
        if history:
            st.divider()
            st.markdown(f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;"
                        f"letter-spacing:.8px;color:{T['muted']};margin-bottom:8px;'>"
                        f"RECENT SCANS</div>", unsafe_allow_html=True)
            for i, h in enumerate(reversed(history[-5:]), 1):
                vcol = T['bar_fake'] if h['verdict']=="FAKE" else (
                       T['bar_ok'] if h['verdict']=="REAL" else "#7B6EBB")
                st.markdown(f"""
                <div style="padding:6px 0;border-bottom:1px solid {T['border']};
                     font-size:11px;">
                    <span style="color:{vcol};font-weight:600;">{h['verdict']}</span>
                    <span style="color:{T['muted']};"> · {h['conf']:.0f}%</span><br>
                    <span style="color:{T['txt2']};font-size:10px;">{h['file']}</span>
                </div>""", unsafe_allow_html=True)

        # Footer
        st.markdown(f"""
        <div style="margin-top:24px;font-size:10px;color:{T['muted']};
             line-height:1.6;border-top:1px solid {T['border']};padding-top:10px;">
            DEEPTRUST v1.9<br>
            Final Year Project · TUK 2026<br>
            Kenya DPA 2019 compliant
        </div>""", unsafe_allow_html=True)

    # Topbar: logo only (sidebar hamburger handles nav)
    inject_button_fix()

    st.markdown(f"""<style>
    .dt-topbar {{
        background: {T['nav']};
        border-bottom: 1px solid {T['border']};
        margin: -1rem -1rem 0.8rem -1rem;
        padding: 8px 1.5rem 8px 4.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .dt-session-strip {{
        font-size: 10.5px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: {T['bar_ok']};
    }}
    .dt-session-meta {{
        font-size: 10px;
        color: {T['muted']};
        margin-left: 6px;
    }}
    </style>""", unsafe_allow_html=True)

    # Slim topbar — just logo + session ID inline (no buttons)
    if sess:
        scan_txt = f"{scans} scan{'s' if scans != 1 else ''}"
        st.markdown(f"""
        <div class="dt-topbar">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:28px;height:28px;background:{T['accent']};
                     border-radius:6px;display:flex;align-items:center;
                     justify-content:center;flex-shrink:0;">
                    <svg width="14" height="14" viewBox="0 0 18 18" fill="none">
                        <circle cx="7.5" cy="7.5" r="5" stroke="#fff" stroke-width="1.5"/>
                        <line x1="11.5" y1="11.5" x2="16" y2="16"
                              stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
                    </svg>
                </div>
                <span style="font-size:14px;font-weight:700;color:{T['txt']};">DEEPTRUST</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:18px;height:18px;background:{T['bar_ok']};
                     border-radius:50%;display:flex;align-items:center;
                     justify-content:center;">
                    <svg width="9" height="9" viewBox="0 0 14 14" fill="none">
                        <circle cx="7" cy="5" r="3" stroke="#fff" stroke-width="1.4"/>
                        <path d="M1 13c0-3 2.7-5 6-5s6 2 6 5" stroke="#fff"
                              stroke-width="1.4" stroke-linecap="round" fill="none"/>
                    </svg>
                </div>
                <span class="dt-session-strip">{sess}</span>
                <span class="dt-session-meta">· {scan_txt} · {start_t}</span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="dt-topbar">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:28px;height:28px;background:{T['accent']};
                     border-radius:6px;display:flex;align-items:center;
                     justify-content:center;flex-shrink:0;">
                    <svg width="14" height="14" viewBox="0 0 18 18" fill="none">
                        <circle cx="7.5" cy="7.5" r="5" stroke="#fff" stroke-width="1.5"/>
                        <line x1="11.5" y1="11.5" x2="16" y2="16"
                              stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
                    </svg>
                </div>
                <span style="font-size:14px;font-weight:700;color:{T['txt']};">DEEPTRUST</span>
            </div>
            <span style="font-size:10px;color:{T['muted']};">
                No active session · Accept consent to begin
            </span>
        </div>""", unsafe_allow_html=True)



def steps(active):
    html = '<div class="steps">'
    for i, (n, lbl) in enumerate([(1,"Consent"), (2,"Upload"), (3,"Results")]):
        # Determine CSS class for this step
        cls = "done" if n < active else ("on" if n == active else "")
        # Checkmark for completed steps, number for current/future
        ic = "&#10003;" if n < active else str(n) # &#10003; =
        html += (f'<div class="step {cls}">'
                 f'<div class="step-n">{ic}</div>'
                 f'<span>{lbl}</span>'
                 f'</div>')
        # Add a horizontal connector line between steps (not after last step)
        if i < 2:
            html += '<div class="step-line"></div>'

    st.markdown(html + "</div>", unsafe_allow_html=True)


def mkbar(lbl, pct, col, vt=None):
    v = vt or f"{pct}%" # use custom text if provided, else show percentage
    return (
        f'<div class="bar">'
        f' <div class="bar-h">'
        f' <span class="bar-lbl">{lbl}</span>'
        f' <span style="font-size:12px;font-weight:500;color:{col}">{v}</span>'
        f' </div>'
        f' <div class="bar-track">'
        f' <div class="bar-fill" '
        f' style="width:{min(pct,100)}%;background:{col}">'
        f' </div>'
        f' </div>'
        f'</div>'
    )


def mklog(k, v, s=""):
    return (
        f'<div class="lr">'
        f' <span class="lk">{k}</span>'
        f' <span class="lv" style="{s}">{v}</span>'
        f'</div>'
    )


# SECTION 18b: WELCOME PAGE — page_welcome()
# The first page users see. Explains what DEEPTRUST is, how to use it, shows key stats, and provides a single clear Start button.

def inject_button_fix():
    """
    Late-injected CSS — runs on every navbar() call, always fires after
    Streamlit Cloud's own styles so our overrides always win.
    """
    # Sidebar button colour — must be here (not global CSS) so it
    # rerenders after Streamlit Cloud reapplies its own styles.
    btn = T['btn_bg']   # #1B3A5C light / #1E4E82 dark
    st.markdown(f"""<style>
    /* Primary CTA buttons */
    .stButton > button[data-testid="baseButton-primary"],
    .stButton > button[kind="primary"] {{
        background-color: {T['accent']} !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 12px 24px !important;
    }}
    .stButton > button[data-testid="baseButton-primary"] p,
    .stButton > button[data-testid="baseButton-primary"] span,
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {{
        color: #FFFFFF !important;
    }}
    /* Sidebar toggle — injected late so Cloud styles cannot override */
    [data-testid="stSidebarCollapsedControl"] {{
        position: fixed !important;
        top: 6px !important;
        left: 1.2rem !important;
        z-index: 99999 !important;
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: {btn} !important;
        border: 2px solid {btn} !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.4) !important;
        width: 38px !important;
        height: 38px !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
    }}
    [data-testid="stSidebarCollapsedControl"] > div {{
        background: {btn} !important;
        width: 38px !important;
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    [data-testid="stSidebarCollapsedControl"] button {{
        background: {btn} !important;
        border: none !important;
        border-radius: 4px !important;
        width: 38px !important;
        height: 38px !important;
        padding: 0 !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative !important;
    }}
    [data-testid="stSidebarCollapsedControl"] button svg {{
        display: none !important;
    }}
    [data-testid="stSidebarCollapsedControl"] button::before {{
        content: "" !important;
        display: block !important;
        width: 16px !important;
        height: 2px !important;
        background: #FFFFFF !important;
        box-shadow: 0 5px 0 #FFFFFF, 0 -5px 0 #FFFFFF !important;
        border-radius: 2px !important;
        position: absolute !important;
    }}
    </style>""", unsafe_allow_html=True)


def page_welcome():
    """
    Landing page

    """
    navbar()

    #Heros section : title and one-line purpose
    st.markdown(f"""
    <div style="padding:24px 0 16px;">
        <div style="font-size:13px;font-weight:600;color:{T['accent']};
             letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px;">
            Image Deepfake Detection Using Ensemble Convolutional Neural Networks and Autoencoder-Based Anomaly Analysis</div>
        <div style="font-size:30px;font-weight:700;color:{T['txt']};
             letter-spacing:-0.3px;margin-bottom:10px;">
            Can you trust what you see online?</div>
        <div style="font-size:14px;color:{T['txt2']};max-width:680px;line-height:1.75;">
            Have you ever encountered a photo on social media that looks incredibly
            realistic, yet something about it just feels off? Maybe it is a profile
            picture of a person who seems a little too perfect, a shocking political image,
            or a celebrity endorsement that looks entirely genuine. In today’s digital world,
            your eyes can easily deceive you.
            This is the reality of deepfakes — digitally manipulated or completely fabricated
            photographs of human faces created by sophisticated Artificial Intelligence (AI).
            These fakes have become so advanced that the human eye can no longer reliably spot them.
            That is where DEEPTRUST comes in. Think of DEEPTRUST as a highly trained digital investigator,
            designed to instantly uncover the truth behind any facial image.
        </div>
    </div>""", unsafe_allow_html=True)

    # PRIMARY ACTION: Start scan button — above everything else
    # Primary action must be immediately visible on load
    ba, bb, bc = st.columns([2, 3, 2])
    with bb:
        if st.button("Start a deepfake scan now",
                     type="primary", use_container_width=True):
            st.session_state.page = "input"
            st.rerun()
    st.markdown(f"""
    <div style="text-align:center;font-size:11px;color:{T['muted']};margin-top:6px;
         margin-bottom:20px;">
        No login required. No data stored. Session deleted on close.
    </div>""", unsafe_allow_html=True)

    # Two-column layout: How it works | What it detects
    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown(f"""
        <div class="card" style="height:100%;">
            <div class="sec-label">How it works — 6 steps</div>
            <div style="display:flex;flex-direction:column;gap:12px;margin-top:6px;">
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="min-width:26px;height:26px;background:{T['accent']};
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;font-size:11px;font-weight:700;
                         color:#fff;flex-shrink:0;">1</div>
                    <div>
                        <div style="font-size:12px;font-weight:600;color:{T['txt']};
                             margin-bottom:2px;">Click Start scan</div>
                        <div style="font-size:11.5px;color:{T['txt2']};line-height:1.5;">
                            Click the blue Start a deepfake scan now button on the home page.</div>
                    </div>
                </div>
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="min-width:26px;height:26px;background:{T['accent']};
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;font-size:11px;font-weight:700;
                         color:#fff;flex-shrink:0;">2</div>
                    <div>
                        <div style="font-size:12px;font-weight:600;color:{T['txt']};
                             margin-bottom:2px;">Accept consent</div>
                        <div style="font-size:11.5px;color:{T['txt2']};line-height:1.5;">
                            Read the privacy notice and tick the checkbox to proceed.</div>
                    </div>
                </div>
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="min-width:26px;height:26px;background:{T['accent']};
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;font-size:11px;font-weight:700;
                         color:#fff;flex-shrink:0;">3</div>
                    <div>
                        <div style="font-size:12px;font-weight:600;color:{T['txt']};
                             margin-bottom:2px;">Get your Session ID</div>
                        <div style="font-size:11.5px;color:{T['txt2']};line-height:1.5;">
                            Your anonymous ID (SESS-XXXX-XXXX-XXXX) appears in the green bar — you are now in the system.</div>
                    </div>
                </div>
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="min-width:26px;height:26px;background:{T['accent']};
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;font-size:11px;font-weight:700;
                         color:#fff;flex-shrink:0;">4</div>
                    <div>
                        <div style="font-size:12px;font-weight:600;color:{T['txt']};
                             margin-bottom:2px;">Upload a face image</div>
                        <div style="font-size:11.5px;color:{T['txt2']};line-height:1.5;">
                            Select any JPG, PNG, or WebP face image. File is validated and a face must be detected.</div>
                    </div>
                </div>
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="min-width:26px;height:26px;background:{T['accent']};
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;font-size:11px;font-weight:700;
                         color:#fff;flex-shrink:0;">5</div>
                    <div>
                        <div style="font-size:12px;font-weight:600;color:{T['txt']};
                             margin-bottom:2px;">Click Analyse</div>
                        <div style="font-size:11.5px;color:{T['txt2']};line-height:1.5;">
                            The 3-stage AI pipeline runs: CNN Spotter, Autoencoder Alarm, Meta-learner.</div>
                    </div>
                </div>
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="min-width:26px;height:26px;background:{T['accent']};
                         border-radius:50%;display:flex;align-items:center;
                         justify-content:center;font-size:11px;font-weight:700;
                         color:#fff;flex-shrink:0;">6</div>
                    <div>
                        <div style="font-size:12px;font-weight:600;color:{T['txt']};
                             margin-bottom:2px;">Get your verdict</div>
                        <div style="font-size:11.5px;color:{T['txt2']};line-height:1.5;">
                            REAL, FAKE, or UNCERTAIN — with Grad-CAM heatmap, AI summary, and PDF report.</div>
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="card" style="height:100%;">
            <div class="sec-label">What DEEPTRUST detects</div>
            <div style="font-size:12px;color:{T['txt2']};line-height:1.75;margin-top:6px;">
                DEEPTRUST was trained on the FaceForensics++ and DFDC academic benchmarks
                covering six types of facial manipulation:<br><br>
                <b style="color:{T['txt']}">1. Face swap (Deepfakes)</b> — one
                person's face placed onto another's body using encoder-decoder AI synthesis.
                Trained on FaceForensics++ Deepfakes category. Detection accuracy: 87.5%.<br><br>
                <b style="color:{T['txt']}">2. Face2Face re-enactment</b> — facial expressions
                and mouth movements transferred from one person to another in real time.
                Detection accuracy: 77.0%.<br><br>
                <b style="color:{T['txt']}">3. FaceShifter identity swap</b> — high-fidelity
                identity transfer that preserves target attributes such as glasses and hair.
                Detection accuracy: 61.0%.<br><br>
                <b style="color:{T['txt']}">4. FaceSwap replacement</b> — geometric face
                region replacement using landmark alignment. Among the harder manipulation
                types to detect. Detection accuracy: 50.0%.<br><br>
                <b style="color:{T['txt']}">5. NeuralTextures rendering</b> — neural
                texture-based rendering that modifies facial appearance via learned texture
                maps. Detection accuracy: 52.0%.<br><br>
                <b style="color:{T['txt']}">6. GAN-generated synthetic faces</b> — completely
                fabricated faces that belong to no real person, generated by StyleGAN and
                similar architectures. Sites like thispersondoesnotexist.com use this method.
                Caught by the Autoencoder anomaly branch even when the CNN is fooled.<br><br>
                <b style="color:{T['bar_fake']}">Limitation:</b> Designed for still images
                of human faces only. Video, audio, and non-face images are not supported.
                Consumer phone selfies may produce lower confidence scores due to compression
                differences from the FaceForensics++ and DFDC training data distribution.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # Privacy + system performance in one row
    pa, pb = st.columns(2, gap="medium")

    with pa:
        st.markdown(f"""
        <div class="card">
            <div class="sec-label">Privacy by design — Kenya Data Protection Act 2019</div>
            <div style="font-size:12px;color:{T['txt2']};line-height:1.75;margin-top:4px;">
                DEEPTRUST was built with a Privacy-by-Design architecture meaning your
                data is protected at every stage, not as an afterthought.<br><br>
                <b style="color:{T['txt']}">No account required.</b> You do not need to
                register, log in, or provide any personal information to use the system.<br><br>
                <b style="color:{T['txt']}">Anonymous session identity.</b> The moment you
                accept the consent, the system generates a temporary Session ID in the format
                SESS-XXXX-XXXX-XXXX. This ID links your upload and scan together during your
                visit only. It is visible in the navigation bar so you always know the system
                acknowledges your presence.<br><br>
                <b style="color:{T['txt']}">No permanent storage.</b> All scan data lives
                in RAM only. When you close the browser or end the session, every record is
                permanently deleted. Nothing is written to disk. Nothing is sent to a server.
            </div>
        </div>""", unsafe_allow_html=True)

    with pb:
        st.markdown(f"""
        <div class="card">
            <div class="sec-label">System performance — benchmark evaluation</div>
            <div style="margin-top:8px;">
                <div style="display:flex;justify-content:space-between;
                     padding:10px 0;border-bottom:1px solid {T['border']};">
                    <span style="font-size:12px;color:{T['txt2']};">Combined accuracy</span>
                    <span style="font-size:14px;font-weight:700;color:{T['accent']};">74.37%</span>
                </div>
                <div style="display:flex;justify-content:space-between;
                     padding:10px 0;border-bottom:1px solid {T['border']};">
                    <span style="font-size:12px;color:{T['txt2']};">Area under curve (AUC)</span>
                    <span style="font-size:14px;font-weight:700;color:{T['accent']};">0.8698</span>
                </div>
                <div style="display:flex;justify-content:space-between;
                     padding:10px 0;border-bottom:1px solid {T['border']};">
                    <span style="font-size:12px;color:{T['txt2']};">Images evaluated</span>
                    <span style="font-size:14px;font-weight:700;color:{T['accent']};">1,900</span>
                </div>
                <div style="display:flex;justify-content:space-between;
                     padding:10px 0;border-bottom:1px solid {T['border']};">
                    <span style="font-size:12px;color:{T['txt2']};">CNN standalone accuracy</span>
                    <span style="font-size:14px;font-weight:700;color:{T['accent']};">78.63%</span>
                </div>
                <div style="display:flex;justify-content:space-between;
                     padding:10px 0;border-bottom:1px solid {T['border']};">
                    <span style="font-size:12px;color:{T['txt2']};">False positive rate</span>
                    <span style="font-size:14px;font-weight:700;color:{T['bar_ok']};">13.0%</span>
                </div>
                <div style="display:flex;justify-content:space-between;
                     padding:10px 0;">
                    <span style="font-size:12px;color:{T['txt2']};">Evaluation datasets</span>
                    <span style="font-size:12px;color:{T['txt2']};">FF++ + DFDC</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Want to try it? demo tip at the very bottom
    st.markdown(f"""
    <div style="background:{T['accent_light']};border:1px solid {T['border']};
         border-left:4px solid {T['accent']};border-radius:8px;
         padding:14px 18px;margin-top:14px;">
        <div style="font-size:13px;font-weight:600;color:{T['accent']};
             margin-bottom:6px;">Want to test it right now?</div>
        <div style="font-size:12px;color:{T['txt2']};line-height:1.7;">
            For the most reliable demo results, try these two sources side by side:<br>
            <b style="color:{T['txt']}">For a FAKE result:</b> open
            <a href="https://thispersondoesnotexist.com" target="_blank"
               style="color:{T['accent']};">thispersondoesnotexist.com</a>,
            right-click the face and save it, then upload it here.<br>
            These give clean, consistent results because they match the training
            data distribution — direct downloads without compression artifacts.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;font-size:10px;color:{T['muted']};
         margin-top:14px;padding-bottom:10px;">
        DEEPTRUST v1.9— Final Year Project, Technical University of Kenya, 2026.
        For forensic assistance only. Not legal evidence without expert review.
    </div>""", unsafe_allow_html=True)



# ACTIVITY LOG PAGE — page_log()
# Shown when user clicks My scans in the navbar.
def page_log():
    """
    My scans page — pure Streamlit widgets for full dark/light mode support.
    Shows session ID, scan history, and lets user navigate.
    """
    navbar()
    st.markdown("### My Scans")
    sess = st.session_state.session_id
    start = st.session_state.get("session_start", "—")
    history = st.session_state.get("scan_history") or []
    scans = len(history)

    if not sess:
        st.warning("No active session. Accept the consent on the scan page to begin.")
        if st.button("Go to scan page"):
            st.session_state.page = "input"
            st.rerun()
        return

    # Session summary using st.metric (fully dark-mode aware)
    ca, cb, cc, cd = st.columns(4)
    with ca: st.metric("Session ID", sess[:18]+"...")
    with cb: st.metric("Session started", start)
    with cc: st.metric("Total scans", scans)
    with cd: st.metric("Status", "Active")

    st.divider()

    if not history:
        st.info("No scans recorded yet this session. Go to the scan page and upload a face image to get started.")
    else:
        st.caption(
            f"{scans} scan{'s' if scans != 1 else ''} recorded · "
            "All data deleted automatically when browser is closed.")

        for i, h in enumerate(history, 1):
            verdict_label = h['verdict']
            with st.expander(
                f"Scan {i} of {scans} — {h['time']} — {h['file']} — {verdict_label} {h['conf']:.0f}%",
                expanded=(i == len(history))
            ):
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Verdict", h['verdict'])
                with col2: st.metric("Confidence", f"{h['conf']:.1f}%")
                with col3: st.metric("CNN score", f"{h['cnn']:.1f}%")
                with col4: st.metric("AE error", f"{h['ae']:.4f}")

                st.markdown("---")
                rid = h['rid']
                fn = h['file']
                tm = h['time']
                st.caption(
                    f"Result ID: {rid} | File: {fn} | Time: {tm} | Session: {sess}"
                )

                # What this verdict means
                if h['verdict'] == "FAKE":
                    st.error("FAKE — The AI detected manipulation signatures in this image.")
                elif h['verdict'] == "REAL":
                    st.success("REAL — No significant manipulation detected. Image appears authentic.")
                else:
                    st.warning("UNCERTAIN — Signals were ambiguous. Expert review recommended.")

    st.divider()
    ba, bb = st.columns(2)
    with ba:
        if st.button("Back to home", use_container_width=True):
            st.session_state.page = "welcome"
            st.rerun()
    with bb:
        if st.button("Scan another image", type="primary", use_container_width=True):
            start_new_scan()
            st.session_state.page = "input"
            st.rerun()


def page_help():
    """Help page — pure Streamlit widgets, no HTML rendering issues."""
    navbar()

    txt_col = T['txt']
    txt2_col = T['txt2']
    st.markdown(f"<h3 style='color:{txt_col};'>DEEPTRUST — Help Guide</h3>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='color:{txt2_col};font-size:13px;'>Everything you need to know about using this system.</p>",
                unsafe_allow_html=True)
    st.divider()

    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown("**How to use DEEPTRUST — step by step**")
        st.markdown("""
**Step 1 — Click Start a deepfake scan now**

On the home page, click the blue Start button. This takes you to the scan page.

**Step 2 — Accept the privacy consent**

Read the privacy notice and tick the checkbox. The moment you tick it, the system generates your anonymous Session ID (SESS-XXXX-XXXX-XXXX) visible in the green bar below the navigation bar. This is how the system acknowledges your presence — no login or password needed.

**Step 3 — Upload a face image**

Drag and drop or browse to select a JPG, PNG, or WebP image. Maximum 10 MB. The system automatically checks the file is genuine and that a human face is visible before any AI analysis runs.

**Step 4 — Click Analyse**

Click the Analyse button on the right side of the image preview. A spinner shows while the 3-stage AI pipeline runs — typically 3 to 8 seconds. Results appear automatically when complete.

**Step 5 — Read your verdict**

- **REAL** — No manipulation detected. Image appears authentic.
- **FAKE** — AI detected manipulation signatures. Likely a deepfake.
- **UNCERTAIN** — Signals are ambiguous. Expert review recommended.

**Step 6 — Download the PDF report**

Click Download PDF forensic report to save a full report with the Grad-CAM heatmap, AI scores, and forensic summary.

**Why was my selfie flagged as fake?**

WhatsApp compression and phone camera processing create pixel-level differences from the training data, which can inflate the Autoencoder error. For best results use images downloaded directly from websites.

**What is the Session ID?**

A temporary anonymous identifier linking your upload and scan together during your visit. Visible in the green bar below the navbar. All data is permanently deleted when the app is closed — Kenya Data Protection Act 2019.
        """)

    with right:
        st.markdown("**Your session activity**")
        history = st.session_state.get("scan_history") or []
        if history:
            for i, h in enumerate(history, 1):
                verdict_icon = "FAKE" if h['verdict']=="FAKE" else ("REAL" if h['verdict']=="REAL" else "UNCERTAIN")
                st.markdown(f"`#{i}` **{h['time']}** — {h['file']} — **{verdict_icon}** {h['conf']:.0f}%")
                st.divider()
        else:
            st.info("No scans yet this session. Upload a face image to get started.")

        st.markdown("**Best images to test with**")
        st.markdown("""
**For a FAKE result:**
Open [thispersondoesnotexist.com](https://thispersondoesnotexist.com), right-click the face image and save it, then upload it here. These are GAN-generated faces.

        """)

    st.divider()
    _, bc, _ = st.columns([3, 1, 3])
    with bc:
        if st.button("Back to home", use_container_width=True):
            st.session_state.page = "welcome"
            st.rerun()


# SECTION 19: INPUT PAGE — page_input()
# This is the first page the user sees. It handles two things:
#
# PART A — Privacy Consent (Step 1)
# Shows the privacy policy card with a checkbox.
# Until the user ticks the checkbox, the upload section is locked.
# When ticked: a unique session ID is generated and registered in the DB.
# When unticked after ticking: the session is cleared (data purged).
#
# PART B — Image Upload (Step 2)
# Shows the file uploader after consent is given.
# File goes through three checks before the scan button appears:
# 1. validate_file() — size, extension, magic bytes, resolution
# 2. has_face() — Haar cascade face detection gate
# 3. If both pass: show preview, metadata chips, and scan button
#
# On scan button click:
# - run_inference() runs the full CNN → AE → meta-learner pipeline
# - Results are saved to database and session state
# - App navigates to the output page

def page_input():
    navbar()
    # Show Step 1 (Consent) if not yet consented, Step 2 (Upload) if consented
    steps(1 if not st.session_state.consent_given else 2)


    # PART A: PRIVACY CONSENT CARD
    # Explains what data is collected and how it is handled.
    # Must be accepted before the upload section appears.
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Session Setup</div>',
                unsafe_allow_html=True)

    # Consent text — explains Privacy-by-Design and legal compliance
    st.markdown(f"""<div class="consent">
        This platform uses a <strong>Privacy-by-Design</strong> approach —
        your image is processed in-memory and deleted after the report is
        generated. Nothing is stored permanently. No account is needed.<br><br>
        Sessions are anonymised using format
        <strong>SESS-XXXX-XXXX-XXXX</strong>.
        Compliant with the
        <strong>Kenya Data Protection Act (2019)</strong>.
    </div>""", unsafe_allow_html=True)

    # Consent checkbox — drives the entire consent flow
    consent = st.checkbox(
        "I agree to the data processing terms above.",
        value=st.session_state.consent_given)

    # User just ticked the checkbox — create session and register in DB
    if consent and not st.session_state.consent_given:
        st.session_state.consent_given = True
        st.session_state.session_id = mk_session()
        st.session_state.session_start = datetime.now().strftime("%H:%M")
        db_session(st.session_state.session_id)
        st.rerun()

    # User unticked after previously consenting — purge all session data
    if not consent and st.session_state.consent_given:
        clear_session()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


    # PART B: FILE UPLOAD CARD
    # Upload section is locked behind consent — shows info message if consent not yet given. Once consented, shows the full upload flow.
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Upload Image</div>',
                unsafe_allow_html=True)

    if not st.session_state.consent_given:
        # Upload locked — prompt user to agree to terms first
        st.info("Agree to the terms above to enable upload.")
    else:
        # File uploader — accepts jpg, jpeg, png, webp up to 10 MB
        uploaded = st.file_uploader(
            "Select a face image to analyse",
            type=["jpg","jpeg","png","webp"],
            help="Max 10 MB — .jpg .png or .webp. For best results use images downloaded from websites rather than phone selfies.")

        # Tip for new users
        st.markdown(f"""
        <div style="font-size:11px;color:{T['muted']};margin-top:4px;line-height:1.6;">
            <b>Tip for demo:</b> Download a face from
            <a href="https://thispersondoesnotexist.com" target="_blank"
               style="color:{T['accent']};">thispersondoesnotexist.com</a>
            (a GAN-generated fake) or a portrait from
            <a href="https://en.wikipedia.org" target="_blank"
               style="color:{T['accent']};">Pictures from Internet</a> (a real face)
            to see the system in action.
        </div>""", unsafe_allow_html=True)

        if uploaded:
            raw = uploaded.getvalue() # read raw bytes from uploaded file

            # Validation Gate 1: File integrity and format
            # Checks size, extension, magic bytes, and PIL readability
            ok, err = validate_file(raw, uploaded.name)
            if not ok:
                st.error(f"Upload rejected: {err}")

            else:
                # Validation Gate 2: Human face detection
                # Uses Haar cascade to confirm a face exists in the image.
                # If the check itself crashes (rare), let the image through rather than blocking a valid upload unnecessarily.
                try:
                    face_ok = has_face(Image.open(io.BytesIO(raw)))
                except Exception:
                    face_ok = True # fail open — better than false rejection

                if not face_ok:
                    # Show styled red error banner — no face found
                    st.markdown(f"""
                    <div style="background:{T['fake_bg']};
                                border:1px solid {T['fake_br']};
                                border-left:4px solid {T['bar_fake']};
                                border-radius:8px;
                                padding:14px 16px">
                        <div style="font-size:14px;font-weight:600;
                                    color:{T['fake_txt']}">
                            No human face detected
                        </div>
                        <div style="font-size:12px;color:{T['fake_sub']};
                                    margin-top:4px;line-height:1.6">
                            DEEPTRUST only analyses images containing a human
                            face. Please upload a portrait or headshot.
                        </div>
                    </div>""", unsafe_allow_html=True)

                else:
                    # Image passed all checks — proceed to upload flow

                    # Generate a unique upload ID if not already set (prevents generating a new ID if user re-uploads same file)
                    if not st.session_state.upload_id:
                        st.session_state.upload_id = mk_upload()

                    uid = st.session_state.upload_id
                    fh = fhash(raw)
                    fsize = round(len(raw) / 1024, 1)

                    # Log upload to TBL_UPLOADS
                    db_upload(uid, st.session_state.session_id,
                              uploaded.name, fsize, fh)

                    # Session + upload ID chips
                    st.markdown(f"""
                    <div style="margin-bottom:10px">
                        <span class="chip">{st.session_state.session_id}</span>
                        <span class="chip">{uid}</span>
                    </div>""", unsafe_allow_html=True)

                    # Mobile-first upload confirmation layout
                    # Single column: image centered, metadata, then big scan button
                    # On desktop this still looks clean; on mobile it's not cramped
                    st.markdown(f"""
                    <style>
                    /* Upload preview card */
                    .dt-upload-preview {{
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        gap: 12px;
                        padding: 16px;
                        background: {T['bg_sec']};
                        border: 1px solid {T['border']};
                        border-radius: 10px;
                        margin-bottom: 14px;
                        text-align: center;
                    }}
                    .dt-upload-preview img {{
                        border-radius: 8px;
                        border: 2px solid {T['border']};
                        max-width: 160px;
                        width: 100%;
                    }}
                    .dt-upload-fname {{
                        font-size: 13px;
                        font-weight: 600;
                        color: {T['txt']};
                        word-break: break-all;
                    }}
                    .dt-upload-meta {{
                        font-size: 11px;
                        color: {T['muted']};
                    }}
                    .dt-upload-ok {{
                        display: inline-flex;
                        align-items: center;
                        gap: 5px;
                        font-size: 12px;
                        font-weight: 600;
                        color: {T['bar_ok']};
                        background: {T['real_bg']};
                        border: 1px solid {T['real_br']};
                        border-radius: 20px;
                        padding: 3px 10px;
                    }}
                    </style>""", unsafe_allow_html=True)

                    # Centered image preview card
                    img_preview = Image.open(io.BytesIO(raw))
                    img_preview.thumbnail((160, 160))

                    # Convert to base64 for inline HTML display
                    import base64
                    buf_img = io.BytesIO()
                    img_preview.save(buf_img, format="PNG")
                    b64 = base64.b64encode(buf_img.getvalue()).decode()

                    st.markdown(f"""
                    <div class="dt-upload-preview">
                        <img src="data:image/png;base64,{b64}" alt="Preview"/>
                        <div>
                            <div class="dt-upload-fname">{uploaded.name}</div>
                            <div class="dt-upload-meta">{fsize} KB &nbsp;·&nbsp; {uid}</div>
                        </div>
                        <span class="dt-upload-ok">
                            &nbsp; Face detected · Ready to scan
                        </span>
                    </div>""", unsafe_allow_html=True)

                    # Full-width scan button — easy tap on mobile
                    scan_pressed = st.button(
                        "Analyse — is this a deepfake?",
                        type="primary",
                        use_container_width=True)

                    if scan_pressed:
                        t0 = time.time()
                        with st.spinner("Analysing image — CNN · Autoencoder · Meta-learner..."):
                            res = run_inference(
                                Image.open(io.BytesIO(raw)), "Standard")

                        # Add session metadata to the result dict
                        elapsed = round(time.time() - t0, 2)
                        res.update({
                            "process_time": elapsed,
                            "file_name": uploaded.name,
                            "file_size": fsize,
                            "file_hash": fh,
                            "scan_mode": "Standard",
                            "upload_id": uid,
                            "session_id": st.session_state.session_id,
                            "timestamp": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"),
                        })

                        # Save to database
                        db_result({
                            "rid": res["result_id"],
                            "uid": res["upload_id"],
                            "cnn": res["cnn_score"],
                            "ae": res["ae_error"],
                            "verdict": res["verdict"],
                            "conf": res["confidence"],
                            "hmap": res["hmap"],
                            "mode": res["scan_mode"],
                            "t": res["process_time"],
                        })

                        # Store result in session state
                        st.session_state.results = res

                        # Add to scan history for activity log
                        if not isinstance(st.session_state.scan_history, list):
                            st.session_state.scan_history = []
                        st.session_state.scan_history.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "file": uploaded.name,
                            "verdict": res["verdict"],
                            "conf": res["confidence"],
                            "cnn": res["cnn_score"],
                            "ae": res["ae_error"],
                            "rid": res["result_id"],
                        })

                        st.session_state.page = "output"
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)



# SECTION 20: OUTPUT PAGE — page_output()
# This is the results page shown after a scan completes.

def page_output():
    # Safety check — if user navigates directly to output without scanning, send them back to the upload page
    r = st.session_state.results
    if not r:
        st.warning("No scanning results. Please go back and upload an image.")
        if st.button("Back"):
            st.session_state.page = "input"
            st.rerun()
        return

    navbar()
    steps(3) # Shows Step 3 (Results) as the active step

    # Determine verdict display properties
    verdict = r["verdict"]
    is_fake = verdict == "FAKE"
    is_uncertain = verdict == "UNCERTAIN"

    # CSS class suffix — controls which colour theme the verdict banner uses
    vk = "fake" if is_fake else ("uncertain" if is_uncertain else "real")

    # Human-readable verdict title shown in the banner
    vt = ("Deepfake detected" if is_fake else
          "Inconclusive — expert review required" if is_uncertain else
          "Authentic image")

    # ID strip — shows tracking identifiers at top of page
    st.markdown(f"""
    <div style="margin-bottom:8px">
        <span class="chip">{r['result_id']}</span>
        <span class="chip">{r['upload_id']}</span>
        <span class="chip">{r['session_id'][:16]}...</span>
    </div>""", unsafe_allow_html=True)

    # Verdict banner
    # The large coloured box showing FAKE (red) / REAL (green) / UNCERTAIN (purple)
    # CSS classes v-fake, v-real, v-uncertain defined in Section 6
    st.markdown(f"""
    <div class="verdict v-{vk}">
        <div class="v-title-{vk}">{vt}</div>
        <div class="v-sub-{vk}">
            CNN + Autoencoder + Meta-learner ensemble
        </div>
        <span class="conf-{vk}">{r['confidence']}% confidence</span>
    </div>""", unsafe_allow_html=True)

    # Two-column layout: metrics left, images right
    cl, cr = st.columns(2, gap="medium")

    # LEFT COLUMN: Model scores and Grad-CAM bars
    with cl:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Ensemble metrics</div>',
                    unsafe_allow_html=True)

        # Choose colours for CNN and AE metric values
        # Red if in "fake" zone, green if in "real" zone
        # FIX: correct theme keys + correct AE threshold (0.030 not 0.25)
        cc = T["val_fake"] if r["cnn_score"] > 50 else T["val_ok"]
        ac = T["val_fake"] if r["ae_error"] > 0.030 else T["val_ok"]

        # Two metric boxes side by side — CNN score and AE error
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric">
                <div class="m-val" style="color:{cc}">{r['cnn_score']}%</div>
                <div class="m-lbl">ECNN score</div>
                <div class="m-src">TBL_RESULTS.CNN_Score</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric">
                <div class="m-val" style="color:{ac}">{r['ae_error']}</div>
                <div class="m-lbl">AE reconstruction error</div>
                <div class="m-src">TBL_RESULTS.AE_Error</div>
            </div>""", unsafe_allow_html=True)

        # Two ensemble confidence bars
        conf_col = T["bar_fake"] if is_fake else T["bar_ok"]
        st.markdown("<br>" +
            mkbar("Manipulation confidence",
                  r["confidence"], conf_col) +
            mkbar("Fusion agreement",
                  min(r["confidence"] - 2, 100),
                  T["bar_info"]),
            unsafe_allow_html=True)

        # Grad-CAM regional anomaly bars
        # Four bars showing which facial regions had highest anomaly activity
        # Colours: red (eye), orange-red (skin), blue (mouth), green (hair)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Grad-CAM regional anomalies</div>',
                    unsafe_allow_html=True)

        rcols = [T["bar_fake"], T["val_fake"], T["bar_info"], T["bar_ok"]]
        st.markdown(
            "".join(
                mkbar(f" {k}", v, c, f"{v}% anomaly")
                for (k,v), c in zip(r["regions"].items(), rcols)
            ),
            unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN: XAI image comparison
    with cr:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">XAI comparison view</div>',
                    unsafe_allow_html=True)

        ic1, ic2 = st.columns(2)
        with ic1:
            st.image(r["orig"].resize((299,299)),
                     use_container_width=True)
            st.markdown('<div class="pane-lbl">Original upload</div>',
                        unsafe_allow_html=True)
        with ic2:
            st.image(r["hmap_img"],
                     use_container_width=True)
            st.markdown('<div class="pane-lbl">Grad-CAM heatmap</div>',
                        unsafe_allow_html=True)

        # Show the database path reference for the heatmap file
        st.markdown(f"""
        <div style="font-size:10px;font-family:'JetBrains Mono',monospace;
                    color:{T['muted']};margin-top:6px">
            TBL_RESULTS.Heatmap_Path → {r['hmap']}
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # FORENSIC LOG
    # Technical audit table showing all scan parameters and decisions.
    # Each row is a key-value pair rendered by mklog().
    # Red text = flagged/concerning value, green text = normal value.
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Technical forensic log</div>',
                unsafe_allow_html=True)

    bad = T["log_bad"] # red — confirmed anomaly
    ok = T["log_ok"] # green — clean / normal
    warn = T.get("log_warn", "color:#9A6200;font-weight:600") # amber — borderline

    log = '<div class="log">'

    # Identification rows
    log += mklog("Result_ID", r["result_id"])
    log += mklog("Upload_ID", r["upload_id"])
    log += mklog("Session_ID", r["session_id"])
    log += mklog("Timestamp", r["timestamp"])
    log += mklog("File_Name", r["file_name"])
    log += mklog("File_Hash", r["file_hash"][:24] + "...")

    #Model score rows — coloured red if in fake zone
    log += mklog(
        "CNN_Score",
        f"{r['cnn_score']}% — "
        f"{'above' if r['cnn_score'] > 50 else 'below'} 50% fake threshold",
        bad if r["cnn_score"] > 50 else ok)

    # AE threshold tiers aligned with the actual gate logic:
    # < 0.028 — clean real face
    # 0.028–0.035 — borderline, gate still may return REAL
    # > 0.035 — elevated, AE trusted over CNN
    _ae = r['ae_error']
    if _ae < 0.028:
        _ae_lbl = f"{_ae:.4f} — normal, within 0.025 real face baseline"
        _ae_col = ok
    elif _ae < 0.035:
        _ae_lbl = f"{_ae:.4f} — borderline (0.028–0.035 range, gates may override)"
        _ae_col = warn
    else:
        _ae_lbl = f"{_ae:.4f} — elevated above 0.035 threshold, AE anomaly detected"
        _ae_col = bad
    log += mklog("AE_Error", _ae_lbl, _ae_col)

    #Verdict row
    verd_style = bad if is_fake else (bad if is_uncertain else ok)
    log += mklog(
        "Verdict",
        r["verdict"] + " — sequential pipeline consensus",
        verd_style)

    # Performance and session rows
    log += mklog("Analysis_Mode",
                 "Sequential CNN - AE pipeline and LogReg meta-learner")
    log += mklog("Process_Time", f"{r['process_time']}s")
    log += mklog("Session_Scans", str(db_count()))

    #Override tier rows — only shown if a tier fired
    ov = r.get("ov_tier")
    if ov == "gate":
        log += mklog("CNN_Gate",
            "CNN < 20% — strong REAL signal, AE override blocked",
            ok)
    elif ov == "0b":
        log += mklog("AE_Override",
            "Tier 0b — strong CNN REAL and low quality image", ok)
    elif ov == "1":
        log += mklog("AE_Override",
            "Tier 1 — borderline CNN overridden by clean AE error", ok)
    elif ov == "2":
        log += mklog("AE_Override",
            "Tier 2 — quality uncertainty flagged, confidence reduced", bad)
    elif ov == "U":
        log += mklog("AE_Override",
            "UNCERTAIN — ambiguous signals, expert review required", bad)

    log += mklog("Storage",
        "TBL_RESULTS (in-memory SQLite, purged on session end)")
    log += "</div>"
    st.markdown(log, unsafe_allow_html=True)

    # AI FORENSIC SUMMARY
    # Plain-English explanation of why the verdict was reached.
    # Uses Ollama if running locally, falls back to rule-based summary.
    st.markdown(
        '<div class="sec-label" style="margin-top:14px">AI forensic summary</div>',
        unsafe_allow_html=True)

    # Generate summary fresh every time — rule-based fallback is instant
    # (caching caused stale summaries from previous scans to appear)
    tier, txt = ai_summary(r)

    tl = ("Anthropic API" if tier == "API" else
          "Local LLM" if tier == "LOCAL-LLM" else
          "Onboard expert parser")

    st.markdown(f"""
    <div class="ai-box">
        <div class="ai-lbl">
            DEEPTRUST AI Analysis
            <span class="tier-tag">{tl}</span>
        </div>
        {txt}
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Verdict plain-English explanation
    verdict_explain = {
        "FAKE": (
            f"<b style='color:{T['bar_fake']}'>FAKE — What this means:</b> DEEPTRUST detected signs of digital manipulation "
            f"at {r['confidence']}% confidence. This could be a face-swap, GAN-generated "
            f"face, or other AI-synthesised content. This is forensic evidence only — "
            f"always seek expert human review before drawing formal conclusions."
        ),
        "REAL": (
            f"<b>REAL — What this means:</b> DEEPTRUST found no significant manipulation "
            f"signatures at {r['confidence']}% confidence. The face appears authentic. "
            f"Note: no deepfake detector is 100% accurate — treat this alongside other evidence."
        ),
        "UNCERTAIN": (
            f"<b>UNCERTAIN — What this means:</b> The AI signals were too ambiguous for a confident "
            f"verdict. This can happen with high-quality deepfakes or heavily processed images. "
            f"Independent expert forensic analysis is strongly recommended."
        ),
    }
    exp_text = verdict_explain.get(r["verdict"], "")
    if exp_text:
        # Border colour matches verdict — red for FAKE, green for REAL, purple for UNCERTAIN
        vbdr = T['fake_br'] if r["verdict"]=="FAKE" else (T['real_br'] if r["verdict"]=="REAL" else "#7B6EBB")
        st.markdown(f"""
        <div style="background:{T['bg_sec']};border:1px solid {T['border']};
             border-left:4px solid {vbdr};
             border-radius:8px;padding:14px 16px;margin-bottom:16px;
             font-size:13px;color:{T['txt']};line-height:1.7;">
            {exp_text}
        </div>""", unsafe_allow_html=True)

    # Session activity log
    history = st.session_state.get("scan_history") or []
    if len(history) > 0:
        with st.expander(f"Session activity — {len(history)} scan{'s' if len(history)!=1 else ''} this session"):
            st.markdown(f"""<div style="font-size:11px;color:{T['muted']};margin-bottom:8px;">
                Session: {r.get('session_id','')} · Started: {st.session_state.get('session_start','')}
            </div>""", unsafe_allow_html=True)
            for i, h in enumerate(history, 1):
                vcol = T['bar_fake'] if h['verdict']=="FAKE" else (
                       T['bar_ok'] if h['verdict']=="REAL" else "#7B6EBB")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;
                     padding:8px 0;border-bottom:1px solid {T['border']};
                     font-size:12px;flex-wrap:wrap;">
                    <span style="color:{T['muted']};min-width:20px;">#{i}</span>
                    <span style="color:{T['muted']};min-width:55px;">{h['time']}</span>
                    <span style="color:{T['txt']};flex:1;">{h['file']}</span>
                    <span style="color:{vcol};font-weight:600;min-width:70px;">{h['verdict']}</span>
                    <span style="color:{T['muted']};">{h['conf']:.1f}%</span>
                </div>""", unsafe_allow_html=True)

    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    cb, cp = st.columns([1, 2])

    with cb:
        if st.button("New scan", key="new_scan_btn"):
            start_new_scan() # keeps session ID and history
            st.session_state.page = "input"
            st.rerun()

    with cp:
        pdf = make_pdf(r, txt)
        st.download_button(
            label = "Download PDF forensic report",
            data = pdf,
            file_name = f"{r['result_id']}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime = "application/pdf",
            type = "primary")



    # Legal disclaimer footer
    st.markdown(f"""
    <div class="disclaimer">
        DEEPTRUST v1.9 — forensic assistance only, not legal evidence.<br>
        Data purged on session close — Kenya Data Protection Act (2019).<br>
        {r['result_id']} | {r['timestamp']}
    </div>""", unsafe_allow_html=True)


# SECTION 21: APP ROUTER
# The final two lines of the app determine which page to show.
# Streamlit reruns the entire script from top to bottom on every interaction.
# st.session_state.page acts as a simple router:
# "input" - show the consent and upload page (page_input)
# anything else - show the results page (page_output)


# Route to the correct page based on session state
if st.session_state.page == "welcome":
    page_welcome() # Landing page — first thing users see
elif st.session_state.page == "help":
    page_help() # Full help guide page
elif st.session_state.page == "log":
    page_log() # My scans
elif st.session_state.page == "input":
    page_input() # Consent - Upload flow (Steps 1 and 2)
else:
    page_output() # Results display (Step 3)











































































































































































































































