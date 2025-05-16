import datetime
import sys
import time
from collections import deque

import config
import db
import fetcher
import model
import strategy

import pandas as pd
from pytorch_forecasting import NHiTS

prod_config = config.get_prod_config()

sys.stderr.write(f"Starting model training at: {datetime.datetime.now(prod_config.eastern)}\n")
best_model_path, train_dataset, val_df=model.train_best_model(prod_config)
sys.stderr.write(f"Completed model training at: {datetime.datetime.now(prod_config.eastern)}\n")
best_model = NHiTS.load_from_checkpoint(best_model_path)
sys.stderr.write(f"Starting model validation at: {datetime.datetime.now(prod_config.eastern)}\n")
val_results = model.rolling_predictions(best_model, val_df, train_dataset, "0", prod_config.context_length, prod_config.prediction_length)
offset = model.analyze(val_results, val_df)
sys.stderr.write(f"Completed model validation at: {datetime.datetime.now(prod_config.eastern)}\n")

now = datetime.datetime.now(prod_config.eastern)
market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
if now < market_open_time:
    time_diff = market_open_time - now
    time.sleep(int(time_diff.total_seconds())-30)

market_open = fetcher.is_market_open(24) #2 minutes worth of retries
if market_open:
    conn, cur = db.connect_to_db()
    fetcher.delete_old_data(conn, cur, prod_config.quote_table)
    cash, stocks = strategy.get_account_data(cur, prod_config.account_table)
    num_buys = 0
    num_sells = 0
    closing_price = 0
    context_window = deque(maxlen=60)
    try:
        time_idx = 0
        # every three seconds, while it is before 1 PM PT / 4PM ET / 9PM GMT(because datetime uses container time which is in GMT)
        while datetime.datetime.now(prod_config.eastern).hour < 16:
            current_data = fetcher.upload_quote_data(prod_config.symbol, conn, cur, prod_config.quote_table)
            current_data.append("0") # default group id for prediction
            current_data.append(time_idx)
            time_idx += 1
            context_window.append(current_data)
            if len(context_window)==context_window.maxlen:
                context_df=pd.DataFrame(list(context_window), columns=prod_config.context_columns)
                context_df = context_df.drop(columns=['timestamp'])
                predicted_price=model.make_prediction(best_model, context_df, train_dataset, prod_config.prediction_length)
                closing_price=context_df.iloc[-1]["currentPrice"]
                cash, stocks, action = strategy.trade(closing_price, predicted_price, cash, stocks, offset)
                if action == "bought":
                    num_buys += 1
                else:
                    num_sells += 1
            time.sleep(3)
        account_value = cash + stocks*closing_price
        strategy.write_account_data(conn, cur, cash, stocks, closing_price, account_value, num_buys, num_sells, prod_config.account_table)
    except Exception as e:
        sys.stderr.write(f"Unexpected error in main loop: {e}\n")
    # after upload finishes, close the connection
    finally:
        db.disconnect_from_db(conn, cur)
        sys.stderr.write("Database connection closed.\n")