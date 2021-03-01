import random
import sys
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

API_ENDPOINT = 'http://model-proxy.lab1-team3.mlops.neu.ro'

### Frontend ###
# image = Image.open("src/app/yelp.png")
image = Image.open("web_ui/yelp.png")

st.image(image, caption="Does the AI think your review positive or negative?", use_column_width=True)

user_input = st.text_area("What are you going to post?")

list = ["My review is positive!", "My review is negative!"]
expectation = st.radio("What do you think?", list)

# UX transformation
user_prediction = 1 if expectation == list[0] else 0  # 1 positive; 0 negative
threshold = 0.5  # Sentiment
today = datetime.today().strftime("%Y-%m-%d")

if st.button("Let me tell you!") and user_input:

    req_data = {'text': user_input, "is_positive_user_answered": user_prediction}
    raw_output = requests.post(f"{API_ENDPOINT}/predictions", json=req_data).json()
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
    try:
        is_positive_model_answered = raw_output["is_positive"]["model_answered"]
        details_mlflow_run_id = raw_output["details"]["mlflow_run_id"]
    except KeyError:
        st.write(f"ERROR: Model raw output: {raw_output}")
        raise

    if user_prediction == is_positive_model_answered:
        st.markdown(':tada:')
        st.write(f"Good job! The model also estimates this review as positive!")
    else:
        st.markdown(':pensive:')
        review = "positive" if is_positive_model_answered else "negative"
        st.write(f"Not quite! The model estimates this review as {review}...")


    try:
        raw_output = requests.get(f"{API_ENDPOINT}/statistics").json()
        # $ curl http://model-proxy.lab1-team3.mlops.neu.ro/statistics
        # {"statistics":{"correctness_rate":0.8}}
        rate = raw_output["statistics"]["correctness_rate"] * 100
        st.write(f"Total model correctness rate: {rate:.2f} %")
    except BaseException as e:
        print(f"Could not load model statistics: {type(e)}: {e}", file=sys.stderr)

    # # TODO: print a fancy statistics
    # try:
    #     resp_data = requests.get(f"{API_ENDPOINT}/predictions").json()
    #     # $ curl http://model-proxy.lab1-team3.mlops.neu.ro/predictions
    #     # [
    #     #  {
    #     #    "text": "somewhat normal cafe",
    #     #    "is_positive_user_answered": false,
    #     #    "is_positive_model_answered": true,
    #     #    "mlflow_run_id": "7398ed28395d46159ae1e31177330688",
    #     #    "inference_elapsed": 0.001,
    #     #    "timestamp": "2021-02-04T01:50:55.407401",
    #     #    "id": 1
    #     #  },
    #     # ...
    #     preds = resp_data
    #     st.write(f"Total: {len(preds)} predictions")
    # except BaseException as e:
    #     print(f"Could not load model predictions: {type(e)}: {e}", file=sys.stderr)

