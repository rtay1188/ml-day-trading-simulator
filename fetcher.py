import yfinance as yf
import time
import psycopg2
import datetime
import pytz
import holidays
import sys

eastern = pytz.timezone("America/New_York")

def is_market_open(max_retries: int = 24) -> bool:
    """
    Check if the U.S. stock market is currently open.

    Args:
        max_retries (int): Number of retries to check if market is open, sleeping 5 seconds between attempts.

    Returns:
        bool: True if market is open, False otherwise.
    """
    nyse_holidays = holidays.NYSE()
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)

    for attempt in range(max_retries):
        now = datetime.datetime.now(eastern)

        if now.weekday() >= 5 or now.date() in nyse_holidays:  # Check if it's a weekend or holiday
            sys.stderr.write("Market is closed (weekend or holiday).\n")
            return False
        elif now.time() < market_open or now.time() > market_close:  # Check trading hours
            sys.stderr.write("Market is closed (off trading hours). Retrying in 5 seconds...\n")
        else:
            sys.stderr.write("Market is open!\n")
            return True
        
        time.sleep(5)  # Wait for 5 seconds before retrying
    
    formatted_datetime = datetime.datetime.now(eastern).strftime("%Y-%m-%d %H:%M:%S")
    sys.stderr.write(f"Max retries reached at {formatted_datetime}. Market may still be closed.\n")
    return False

def upload_quote_data(symbol: str, conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor, table_name: str) -> list:
    """
    Fetch current stock quote data for a given symbol using yfinance, upload it to the database,
    and return the values.

    Args:
        symbol (str): The stock ticker symbol.
        conn (connection): The PostgreSQL database connection.
        cur (cursor): The PostgreSQL database cursor.
        table_name (str): The table name.

    Returns:
        list[any]: List of inserted values or [] on failure.
    """    
    expected_keys = ['previousClose', 'open', 'dayLow', 'dayHigh', 'regularMarketPreviousClose', 'regularMarketOpen', 'regularMarketDayLow', 'regularMarketDayHigh', 'dividendRate', 'dividendYield', 'payoutRatio', 'beta', 'trailingPE', 'forwardPE', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 'averageDailyVolume10Day', 'bid', 'ask', 'bidSize', 'askSize', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months', 'fiftyDayAverage', 'twoHundredDayAverage', 'profitMargins', 'sharesPercentSharesOut', 'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio', 'shortPercentOfFloat', 'earningsQuarterlyGrowth', 'trailingEps', 'forwardEps', '52WeekChange', 'SandP52WeekChange', 'lastDividendValue', 'currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice', 'recommendationMean']
    try:
        ticker = yf.Ticker(symbol)
        quote = ticker.get_info()
        values = [datetime.datetime.now(eastern)] + list(map(quote.get, expected_keys))
        
        columns = ['timestamp'] + expected_keys  # Add timestamp column
        values_placeholders = ', '.join(['%s'] * len(columns))  # Placeholder for values

        insert_query = f"""
        INSERT INTO {table_name} ({', '.join([f'"{col}"' for col in columns])})
        VALUES ({values_placeholders});
        """
        try:
            cur.execute(insert_query, values)
            conn.commit()
        except psycopg2.Error as e:
            sys.stderr.write(f"Error while inserting to db: {e}\n")
        return values
    except Exception as e:
        sys.stderr.write(f"Exception occured while getting quote data: {e}\n")
        return []

def delete_old_data(conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor, table_name: str)->None:
    """
    Delete quote data entries older than some weeks from the database.

    Args:
        conn (connection): The PostgreSQL database connection.
        cur (cursor): The PostgreSQL database cursor.
        table_name (str): Name of table to delete old data from.

    Returns:
        None
    """
    try:
        sys.stderr.write("Running delete_old_data()\n")
        weeks_ago = datetime.datetime.now(eastern) - datetime.timedelta(weeks=5)
        timestamp_column = "timestamp"
        cur.execute(f"""
            DELETE FROM {table_name}
            WHERE {timestamp_column} < %s
        """, (weeks_ago,))
        sys.stderr.write(f"Deleted {cur.rowcount} rows.\n")
        conn.commit()
    except Exception as e:
        sys.stderr.write(f"Exception occured while deleting old data: {e}\n")