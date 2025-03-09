import streamlit as st
import pandas as pd
import openai

st.write(st.secrets)  # This will print all available secrets to the app
openai.api_key = st.secrets["openai_api_key"]
