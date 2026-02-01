import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="Spotify Study Recommender",
    page_icon="🎧",
    layout="centered"
)

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
        # Placeholder for your recommender logic
        st.success(f"Generating playlist for **{subject}**")
        st.write(f"- Focus level: {focus_level}")
        st.write(f"- Mood: {mood}")
        st.write(f"- Energy: {energy}")

        st.info("Spotify recommendation logic goes here 👀")
