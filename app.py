from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_engagement_tree_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"

st.set_page_config(
    page_title="Gaming Engagement Predictor",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 Gaming Engagement Predictor")
st.write(
    "Predict a player's engagement category using the tuned Decision Tree model. "
    "Enter the player's details below to get a prediction."
)

if not MODEL_PATH.exists():
    st.error("The trained model file was not found.")
    st.info(
        "First add the dataset to `data/online_gaming_behavior_dataset.csv`, "
        "then run `python train_model.py` to create the model file."
    )
    st.stop()

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

if METRICS_PATH.exists():
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
else:
    metrics = {"test_accuracy": 0.9023354564755839, "macro_f1": 0.897715}

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Notebook test accuracy", f"{metrics.get('test_accuracy', 0.9023) * 100:.2f}%")
metric_col2.metric("Macro F1-score", f"{metrics.get('macro_f1', 0.8977) * 100:.2f}%")
metric_col3.metric("Algorithm", "Decision Tree")
st.caption("Reported notebook result: 90.23% test accuracy. Your locally retrained result may vary if the dataset changes.")

st.divider()
st.subheader("Player information")

with st.form("prediction_form"):
    left, middle, right = st.columns(3)

    with left:
        age = st.number_input("Age", min_value=1, max_value=100, value=26, step=1)
        gender = st.text_input("Gender", value="Male", help="Use a category present in the training dataset.")
        location = st.text_input("Location", value="USA", help="Use a category present in the training dataset.")
        game_genre = st.text_input("Game genre", value="Action", help="Use a category present in the training dataset.")

    with middle:
        play_time = st.number_input("Play time (hours)", min_value=0.0, max_value=1000.0, value=7.5, step=0.5)
        in_game_purchases = st.selectbox("In-game purchases", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        game_difficulty = st.text_input("Game difficulty", value="Medium", help="Use a category present in the training dataset.")
        sessions_per_week = st.number_input("Sessions per week", min_value=0, max_value=500, value=10, step=1)

    with right:
        avg_session_duration = st.number_input("Average session duration (minutes)", min_value=0.0, max_value=10000.0, value=90.0, step=5.0)
        player_level = st.number_input("Player level", min_value=0, max_value=100000, value=22, step=1)
        achievements = st.number_input("Achievements unlocked", min_value=0, max_value=100000, value=15, step=1)

    submitted = st.form_submit_button("Predict engagement", type="primary", use_container_width=True)

if submitted:
    player = pd.DataFrame([{
        "Age": age,
        "Gender": gender.strip(),
        "Location": location.strip(),
        "GameGenre": game_genre.strip(),
        "PlayTimeHours": play_time,
        "InGamePurchases": in_game_purchases,
        "GameDifficulty": game_difficulty.strip(),
        "SessionsPerWeek": sessions_per_week,
        "AvgSessionDurationMinutes": avg_session_duration,
        "PlayerLevel": player_level,
        "AchievementsUnlocked": achievements,
    }])

    if any(not player[col].iloc[0] for col in ["Gender", "Location", "GameGenre", "GameDifficulty"]):
        st.warning("Please complete all categorical fields.")
    else:
        prediction = model.predict(player)[0]
        st.divider()
        st.subheader("Prediction result")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(player)[0]
            classes = model.classes_
            probability_df = pd.DataFrame({"Engagement level": classes, "Probability": probabilities})
            probability_df = probability_df.sort_values("Probability", ascending=False)
            confidence = float(probabilities[list(classes).index(prediction)])
        else:
            probability_df = pd.DataFrame()
            confidence = None

        result_col, chart_col = st.columns([1, 2])
        with result_col:
            st.success(f"Predicted engagement: **{prediction}**")
            if confidence is not None:
                st.metric("Model probability for this class", f"{confidence * 100:.1f}%")
        with chart_col:
            if not probability_df.empty:
                fig = px.bar(
                    probability_df,
                    x="Engagement level",
                    y="Probability",
                    color="Engagement level",
                    text=probability_df["Probability"].map(lambda x: f"{x:.1%}"),
                    title="Prediction probabilities",
                    range_y=[0, 1],
                )
                fig.update_layout(showlegend=False, yaxis_title="Probability", xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("View submitted player data"):
            st.dataframe(player, use_container_width=True, hide_index=True)

st.divider()
st.caption("Educational ML demonstration. Predictions are estimates from the trained dataset and should not be treated as certainty.")
