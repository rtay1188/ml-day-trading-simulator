# ml-day-trading-simulator

This day trading simulator uses a [Neural Hierarchical Interpolation for Time Series](https://arxiv.org/abs/2201.12886) (NHITS) model to simulate high-ish frequency day trading. 

## Overview

1. It uses [yfinance](https://github.com/ranaroussi/yfinance) to poll stock data every 3 seconds.
2. It stores that data in a [PostgreSQL](https://www.postgresql.org/) database.
3. Before the market opens each day, it pulls the data and trains a new NHITS model.
4. In addition to polling the stock data every 3 seconds and writing to the database, it uses the model to predict the stock price 3 seconds later.
5. Based on the predicted price, it simulates a buy or sell and tracks a simulated account balance and stock holdings.
6. At the end of the day, it record the account's value and trades executed throughout the day.

## Setup

1. Setup a PostgreSQL database with the quote_data and account_data table columns set as the same in config.py.
2. Provide valid database connection parameters in db.py.
3. Build a docker image with the provided dockerfile and run as a container OR use notebook.ipynb to run the code manually.
4. Setup a cronjob to run the container before each weekday.

## Common Issues
1. Yahoo Finance often rate limits quote info pulls. This is the main blocker for this script and often results in lack of data.