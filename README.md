# � Spotify Study Recommender (webpage)

This repository contains a single-page Streamlit web app that generates music recommendations tailored to a study subject, desired focus level, mood, and energy level. The recommender uses audio feature columns (e.g., energy, tempo, valence, instrumentalness, speechiness) from a CSV dataset to build a target feature vector and ranks tracks by cosine similarity.

Features
- Enter a study subject (e.g. "Linear Algebra") and choose focus, mood and energy to produce a ranked playlist.
- Dataset-driven: reads CSV(s) from `data/raw/` (see "Dataset placement" below).
- Uses Streamlit caching decorators (`@st.cache_data`, `@st.cache_resource`) to speed up repeated loads and expensive computations.

Quick start (macOS / zsh)

1. Create and activate a virtual environment in the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run streamlit_app.py
```

Dataset placement
- Place your CSV dataset(s) in `data/raw/`. The app looks for `dataset.csv` or `spotify_tracks.csv` by default (see `DATA_PATHS` in `streamlit_app.py`).
- The app coerces feature columns to numeric and drops rows with missing audio features. If your CSV uses different column names, update `FEATURE_COLS_CANDIDATES`, `TRACK_NAME_CANDIDATES`, and `ARTIST_CANDIDATES` at the top of `streamlit_app.py`.

How to use the webpage
- Fill in the "What are you studying?" field and adjust the sliders/select boxes for focus, mood, and energy.
- Click "Generate Recommendations" to load the dataset and show a ranked list (match score and a few feature columns are shown).

Notes for contributors
- Keep UI and data-loading logic in `streamlit_app.py` unless extracting a small module for reuse.
- Preserve the `@st.cache_*` decorators when refactoring caching-critical functions.
