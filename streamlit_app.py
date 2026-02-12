import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


# ---------- Page config ----------
st.set_page_config(
    page_title="Spotify Study Recommender",
    page_icon="🎧",
    layout="centered"
)

# ---------- Dataset config ----------
DATA_PATHS = [
    Path("data/raw/dataset.csv"),
    Path("data/raw/spotify_tracks.csv"),
]

# These are the audio features we’ll use for similarity
FEATURE_COLS_CANDIDATES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness", "speechiness"
]

TRACK_NAME_CANDIDATES = ["track_name", "name", "track"]
ARTIST_CANDIDATES = ["artists", "artist_name", "artist"]

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _find_dataset_path() -> Path:
    for p in DATA_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset CSV not found. Put dataset.csv in data/raw/ (or update DATA_PATHS)."
    )

@st.cache_data
def load_tracks() -> pd.DataFrame:
    csv_path = _find_dataset_path()
    df = pd.read_csv(csv_path)

    # Identify track + artist columns (best effort)
    track_col = _pick_col(df, TRACK_NAME_CANDIDATES)
    artist_col = _pick_col(df, ARTIST_CANDIDATES)

    # Keep only columns we need (only those that exist)
    feature_cols = [c for c in FEATURE_COLS_CANDIDATES if c in df.columns]
    keep_cols = []
    if track_col: keep_cols.append(track_col)
    if artist_col: keep_cols.append(artist_col)
    keep_cols += feature_cols

    if not feature_cols:
        raise ValueError(
            f"None of the expected feature columns were found. "
            f"Expected something like: {FEATURE_COLS_CANDIDATES}. "
            f"Your columns are: {list(df.columns)[:40]} ..."
        )

    df = df[keep_cols].copy()

    # Coerce features to numeric and drop bad rows
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols)

    # Basic de-dup if possible
    if track_col and artist_col:
        df = df.drop_duplicates(subset=[track_col, artist_col])

    # Store resolved names for later use
    df.attrs["track_col"] = track_col
    df.attrs["artist_col"] = artist_col
    df.attrs["feature_cols"] = feature_cols
    df.attrs["csv_path"] = str(_find_dataset_path())
    return df

@st.cache_resource
def fit_scaler_and_matrix(df: pd.DataFrame):
    feature_cols = df.attrs["feature_cols"]
    X = df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return scaler, Xs

def build_target(df: pd.DataFrame, subject: str, focus_level: int, mood: str, energy_level: str) -> np.ndarray:
    feature_cols = df.attrs["feature_cols"]
    stats_mean = df[feature_cols].mean()
    stats_std = df[feature_cols].std().replace(0, 1e-9)
    stats_min = df[feature_cols].min()
    stats_max = df[feature_cols].max()

    # Start at population mean
    target = stats_mean.copy()

    # Focus: higher focus -> more instrumental, less speechy, slightly less energetic
    focus = (focus_level - 5.5) / 4.5  # roughly -1..+1
    if "instrumentalness" in feature_cols:
        target["instrumentalness"] += 0.50 * focus * stats_std["instrumentalness"]
    if "speechiness" in feature_cols:
        target["speechiness"] -= 0.50 * focus * stats_std["speechiness"]
    if "energy" in feature_cols:
        target["energy"] -= 0.15 * focus * stats_std["energy"]

    # Mood tweaks
    mood = mood.lower()
    if mood == "calm":
        if "energy" in feature_cols: target["energy"] -= 0.35 * stats_std["energy"]
        if "tempo" in feature_cols: target["tempo"] -= 0.35 * stats_std["tempo"]
        if "valence" in feature_cols: target["valence"] += 0.10 * stats_std["valence"]
    elif mood == "upbeat":
        if "energy" in feature_cols: target["energy"] += 0.35 * stats_std["energy"]
        if "tempo" in feature_cols: target["tempo"] += 0.30 * stats_std["tempo"]
        if "valence" in feature_cols: target["valence"] += 0.35 * stats_std["valence"]
    elif mood == "dark":
        if "valence" in feature_cols: target["valence"] -= 0.55 * stats_std["valence"]
        if "energy" in feature_cols: target["energy"] += 0.05 * stats_std["energy"]
    elif mood == "ambient":
        if "instrumentalness" in feature_cols: target["instrumentalness"] += 0.50 * stats_std["instrumentalness"]
        if "acousticness" in feature_cols: target["acousticness"] += 0.25 * stats_std["acousticness"]
        if "speechiness" in feature_cols: target["speechiness"] -= 0.35 * stats_std["speechiness"]
        if "energy" in feature_cols: target["energy"] -= 0.20 * stats_std["energy"]
        if "tempo" in feature_cols: target["tempo"] -= 0.20 * stats_std["tempo"]
    # Neutral: no extra change

    # Energy level selector
    e = energy_level.lower()
    if e == "low":
        if "energy" in feature_cols: target["energy"] -= 0.35 * stats_std["energy"]
        if "tempo" in feature_cols: target["tempo"] -= 0.20 * stats_std["tempo"]
    elif e == "high":
        if "energy" in feature_cols: target["energy"] += 0.35 * stats_std["energy"]
        if "tempo" in feature_cols: target["tempo"] += 0.20 * stats_std["tempo"]
    # Medium: no extra change

    # Very light subject heuristic (Week 4 friendly)
    s = subject.lower()
    stem_keywords = ["math", "algebra", "calculus", "physics", "cs", "coding", "program", "linear", "statistics"]
    reading_keywords = ["history", "english", "writing", "reading", "essay", "grammar", "literature"]

    if any(k in s for k in stem_keywords):
        if "speechiness" in feature_cols: target["speechiness"] -= 0.25 * stats_std["speechiness"]
        if "instrumentalness" in feature_cols: target["instrumentalness"] += 0.25 * stats_std["instrumentalness"]
    elif any(k in s for k in reading_keywords):
        if "energy" in feature_cols: target["energy"] -= 0.15 * stats_std["energy"]
        if "acousticness" in feature_cols: target["acousticness"] += 0.15 * stats_std["acousticness"]

    # Clip to observed ranges so it stays realistic
    target = target.clip(stats_min, stats_max)

    return target.to_numpy(dtype=float)

def recommend(df: pd.DataFrame, subject: str, focus_level: int, mood: str, energy_level: str, top_n: int = 20) -> pd.DataFrame:
    scaler, Xs = fit_scaler_and_matrix(df)
    target_vec = build_target(df, subject, focus_level, mood, energy_level).reshape(1, -1)
    target_s = scaler.transform(target_vec)

    sims = cosine_similarity(Xs, target_s).ravel()
    top_idx = np.argsort(-sims)[:top_n]

    track_col = df.attrs["track_col"]
    artist_col = df.attrs["artist_col"]
    feature_cols = df.attrs["feature_cols"]

    show_cols = []
    if track_col: show_cols.append(track_col)
    if artist_col: show_cols.append(artist_col)
    # show a few features for transparency
    for c in ["energy", "tempo", "valence", "instrumentalness", "speechiness"]:
        if c in feature_cols and c not in show_cols:
            show_cols.append(c)

    out = df.iloc[top_idx][show_cols].copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.insert(1, "match_score", sims[top_idx])
    return out.reset_index(drop=True)

# ---------- Title & description ----------
st.title("🎧 Spotify Study Recommender")
st.write(
    "Generate music recommendations tailored to your study subject, "
    "mood, and focus level."
)

st.divider()

# ---------- User inputs ----------
st.header("📚 Study Context")

subject = st.text_input(
    "What are you studying?",
    placeholder="e.g. Linear Algebra, Tamil grammar, History"
)

focus_level = st.slider(
    "How focused do you want to be?",
    min_value=1,
    max_value=10,
    value=7
)

mood = st.selectbox(
    "Preferred mood",
    ["Neutral", "Calm", "Upbeat", "Dark", "Ambient"]
)

energy = st.selectbox(
    "Energy level",
    ["Low", "Medium", "High"]
)

st.divider()

# ---------- Action button ----------
if st.button("🎶 Generate Recommendations"):
    st.subheader("Your recommendations")

    if subject.strip() == "":
        st.warning("Please enter a study subject.")
    else:
        try:
            df = load_tracks()
            st.caption(f"Dataset loaded from: {Path(df.attrs['csv_path']).name} | Rows: {len(df):,}")

            recs = recommend(df, subject=subject, focus_level=focus_level, mood=mood, energy_level=energy, top_n=20)

            st.success(f"Generated a playlist for **{subject}**")
            st.dataframe(recs, hide_index=True)

        except Exception as e:
            st.error(f"Could not generate recommendations: {e}")

