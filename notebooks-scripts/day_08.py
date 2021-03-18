import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go   

import datetime

container_1 = st.beta_container()
container_2 = st.beta_container()
sidebar = st.beta_container()


def get_data(ticker,start,end,interval):

    t = yf.Ticker(ticker)

    return t.history(start=start, end=end, interval=interval)


with container_1:
	st.title('Stock comparison')

with sidebar:
    tickers = ['AAPL','TSLA','FB']

    stock_dropdown_1 = st.sidebar.selectbox(label="First stock", options=tickers)
    stock_dropdown_2 = st.sidebar.selectbox(label="Second stock", options=tickers)

    today = datetime.date.today()
    two_years_ago = today - datetime.timedelta(days=730)
    start_date = st.sidebar.date_input('Start date', two_years_ago)
    end_date = st.sidebar.date_input('End date', today)

    interval = ['1d','1m']

    interval_dropdown = st.sidebar.selectbox(label="Interval", options=interval)


with container_2:
    history_1 = get_data(stock_dropdown_1,start_date,end_date,interval_dropdown)
    history_2 = get_data(stock_dropdown_2,start_date,end_date,interval_dropdown)

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=history_1.index, y=history_1['Close'], name=stock_dropdown_1))
    fig.add_trace(go.Scatter(x=history_2.index, y=history_2['Close'], name=stock_dropdown_2))
    
    st.plotly_chart(fig)