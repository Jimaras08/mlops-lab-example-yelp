import random
from datetime import datetime
import requests

import streamlit as st
import random
from PIL import Image
from sqlalchemy import create_engine

### Backend ###
#db = create_engine(
#    "postgres://zbdsjugqjjajzs:67e43133445a6275d249d85bfaca56076f79d410a93050da65dcc585df747498@ec2-54-246-89-234.eu-west-1.compute.amazonaws.com:5432/db609fft9r2h1t"
#)

API_ENDPOINT = 'http://model-proxy.lab1-team3.neu.ro/predict'

### Frontend ###
image = Image.open("src/app/yelp.png")
# image = Image.open("yelp.png")

st.image(image, caption="Does the AI think your review positive or negative?", use_column_width=True)

user_input = st.text_area("What are you going to post?")

list = ["My review is positive!", "My review is negative!"]
expectation = st.radio("What do you think?", list)

# UX transformation
user_prediction = 1 if expectation == list[0] else 0  # 1 positive; 0 negative
threshold = 0.5  # Sentiment
today = datetime.today().strftime("%Y-%m-%d")

if st.button("Let me tell you!") and user_input:

    sentiment = requests.post(API_ENDPOINT, {'text': user_input}).model_output_is_positive

    if user_prediction == sentiment:
        st.write(
            f"Good job! The AI estimates this review is positive."
        )
    else:
        st.write(
            f"Not quite! The AI estimates this review is positive with probability."
        )

    """
    db.execute(
        f"INSERT INTO stats VALUES ('{today}', '{user_input}', {model_prediction}, {sentiment}, {user_prediction})"
    )
    true_predictions = db.execute(
        "SELECT count(*) FROM stats where sentiment = user_prediction"
    ).fetchone()["count"]
    total_predictions = db.execute("SELECT count(*) FROM stats").fetchone()["count"]
    st.write(
        f"Cool, we have now {round(100*true_predictions/total_predictions)} % correct predictions!"
    )
    st.write(f"Number of total predictions: {total_predictions}.")
    """