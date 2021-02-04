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

    # See REST API instructions here: https://github.com/dmangonakis/Yelp_Dataset/pull/20

    # NOTE!!! Model API URI will change today from 'http://34.91.75.82' to 'http://model-proxy.lab1-team3.neu.ro'

    sentiment = requests.post(API_ENDPOINT, {'text': user_input}).model_output_is_positive
    # To do inference:
    # $ curl -X POST -H "Content-Type: application/json" \
    #   -d '{"text": "awful disguisting cafe", "is_positive_user_answered": false}' \
    #   http://34.91.75.82/predictions | jq
    # {
    #   "id": 7,
    #   "text": "awful disguisting cafe",
    #   "is_positive": {
    #     "user_answered": false,
    #     "model_answered": false
    #   },
    #   "details": {
    #     "mlflow_run_id": "3acade02674549b19044a59186d97db4",
    #     "inference_elapsed": 0.0010228157043457031,
    #     "timestamp": "2021-02-04T06:15:36.160586"
    #   }
    # }

    if user_prediction == sentiment:
        st.write(
            f"Good job! The AI estimates this review is positive."
        )
    else:
        st.write(
            f"Not quite! The AI estimates this review is positive with probability."
        )

    # To get statistics:
    # $ curl http://34.91.75.82/statistics
    # {"statistics":{"correctness_rate":0.8}}
