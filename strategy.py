import psycopg2
import db
import datetime
import sys
import pytz
eastern = pytz.timezone("America/New_York")

def trade(currentPrice: float, predictedPrice: float, currentCash: float, currentStock: int, offset: float) -> tuple[float, int, str]:
    """
    Makes a trading decision based on predicted vs. current price.

    - If predicted price is greater than or equal to current price, buys as much stock as possible.
    - If predicted price is less than current price, sells all current stock.

    Args:
        currentPrice (float): The current price of the stock.
        predictedPrice (float): The predicted future price of the stock.
        currentCash (float): The current cash balance.
        currentStock (int): The number of stocks currently held.
        offset (float): Correction value for predictedPrice based on validation results.

    Returns:
        tuple[float, int, str]: Updated (cash balance, stock count, ["bought"/"sold"]).
    """
    action="bought"
    predictedPrice += offset
    if predictedPrice >= currentPrice:
        # Buy as much stock as possible
        num_to_buy = int(currentCash // currentPrice)
        currentCash -= num_to_buy * currentPrice
        currentStock += num_to_buy
    else:
        # Sell all stock
        currentCash += currentStock * currentPrice
        currentStock = 0
        action="sold"
    return currentCash, currentStock, action

def get_account_data(cur: psycopg2.extensions.cursor, table_name: str)->tuple[float, int]:
    """
    Retrieves the most recent account state from the specified PostgreSQL table.

    Args:
        cur (psycopg2.extensions.cursor): The database cursor used to execute the query.
        table_name (str): The name of the table containing account data.

    Returns:
        tuple[float, int]: A tuple containing:
            - cash (float): The latest recorded cash balance.
            - numShares (int): The latest recorded number of owned shares.

    Raises:
        IndexError: If the table is empty or no rows are returned.
        KeyError: If expected columns are not present in the table.
    """
    overall_account_data=db.fetch_data(cur, table_name)
    last_row=overall_account_data.iloc[-1]
    return last_row["cash"], last_row["numShares"]

def write_account_data(conn: psycopg2.extensions.connection, cur: psycopg2.extensions.cursor, current_cash: float, current_stock: int, closing_price: float, account_value: float, buys: int, sells: int, table_name: str)->None:
    """
    Inserts a new record of account data into the specified PostgreSQL table.

    Args:
        conn (psycopg2.extensions.connection): The active database connection.
        cur (psycopg2.extensions.cursor): The database cursor used for executing the insert query.
        current_cash (float): Current cash balance.
        current_stock (int): Current number of shares held.
        closing_price (float): The latest stock closing price.
        account_value (float): The total value of the account (cash + value of shares).
        buys (int): Number of buy transactions since last update.
        sells (int): Number of sell transactions since last update.
        table_name (str): The name of the table to insert the account data into.

    Returns:
        None

    Logs:
        Writes error messages to stderr if database operations fail.
    """
    columns = ['timestamp', 'cash', 'numShares', 'closingPrice', 'accountValue', 'buys', 'sells']
    try:
        values = [datetime.datetime.now(eastern), float(current_cash), current_stock, float(closing_price), float(account_value), buys, sells]
        values_placeholders = ', '.join(['%s'] * len(columns))  # Placeholder for values
        print(values)
        insert_query = f"""
        INSERT INTO {table_name} ({', '.join([f'"{col}"' for col in columns])})
        VALUES ({values_placeholders});
        """
        try:
            cur.execute(insert_query, values)
            conn.commit()
        except psycopg2.Error as e:
            sys.stderr.write(f"Error while inserting to account db: {e}\n")
    except Exception as e:
        sys.stderr.write(f"Exception occured while writing account data: {e}\n")
