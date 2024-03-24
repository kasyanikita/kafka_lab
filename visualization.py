import streamlit as st 
import json
from kafka_classes.consumer import MyConsumer
from config import bootstrap_servers, metrics_topic
import altair as alt
import pandas as pd

st.set_page_config(
    page_title="Smiling Classification Dashboard",
    layout="wide",
)
consumer = MyConsumer(bootstrap_servers, metrics_topic)
    
st.session_state["train_loss"] = []
st.session_state["test_loss"] = []
st.session_state["train_accuracy"] = []
st.session_state["test_accuracy"] = []

train_loss_chart = st.empty()
test_loss_chart = st.empty()
train_accuracy_chart = st.empty()
test_accuracy_chart = st.empty()

while True:
    stock_data = consumer.poll()
    st.session_state["train_loss"].append(stock_data['train_loss'])
    st.session_state["test_loss"].append(stock_data['test_loss'])
    st.session_state["train_accuracy"].append(stock_data['train_accuracy'])
    st.session_state["test_accuracy"].append(stock_data['test_accuracy'])

    df = pd.DataFrame({"train_loss": st.session_state["train_loss"],
                    "test_loss": st.session_state["test_loss"],
                    "train_accuracy": st.session_state["train_accuracy"],
                    "test_accuracy": st.session_state["test_accuracy"],
                    "iterations": [i for i in range(len(st.session_state["train_loss"]))]})

    train_loss_chart.altair_chart(
        alt.Chart(df).mark_line(point=True).encode(x="iterations", y="train_loss")
                                        .properties(title ="Train loss"), use_container_width=True)
    test_loss_chart.altair_chart(
        alt.Chart(df).mark_line(point=True).encode(x="iterations", y="test_loss")
                                        .properties(title ="Test loss"), use_container_width=True)
    train_accuracy_chart.altair_chart(
        alt.Chart(df).mark_line(point=True).encode(x="iterations", y="train_accuracy")
                                        .properties(title ="Train accuracy"), use_container_width=True)
    test_accuracy_chart.altair_chart(
        alt.Chart(df).mark_line(point=True).encode(x="iterations", y="test_accuracy")
                                        .properties(title ="Test accuracy"), use_container_width=True)