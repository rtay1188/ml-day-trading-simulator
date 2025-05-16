import pytz

class BaseConfig:
    def __init__(self):
        self.eastern = pytz.timezone("America/New_York")
        self.symbol = "aapl"
        self.context_length = 60
        self.prediction_length = 15
        self.context_columns = ['timestamp', 'previousClose', 'open', 'dayLow', 'dayHigh', 'regularMarketPreviousClose', 'regularMarketOpen', 'regularMarketDayLow', 'regularMarketDayHigh', 'dividendRate', 'dividendYield', 'payoutRatio', 'beta', 'trailingPE', 'forwardPE', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 'averageDailyVolume10Day', 'bid', 'ask', 'bidSize', 'askSize', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months', 'fiftyDayAverage', 'twoHundredDayAverage', 'profitMargins', 'sharesPercentSharesOut', 'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio', 'shortPercentOfFloat', 'earningsQuarterlyGrowth', 'trailingEps', 'forwardEps', '52WeekChange', 'SandP52WeekChange', 'lastDividendValue', 'currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice', 'recommendationMean', 'group_id', 'time_idx']
        self.quote_keys = ['previousClose', 'open', 'dayLow', 'dayHigh', 'regularMarketPreviousClose', 'regularMarketOpen', 'regularMarketDayLow', 'regularMarketDayHigh', 'dividendRate', 'dividendYield', 'payoutRatio', 'beta', 'trailingPE', 'forwardPE', 'volume', 'regularMarketVolume', 'averageVolume', 'averageVolume10days', 'averageDailyVolume10Day', 'bid', 'ask', 'bidSize', 'askSize', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months', 'fiftyDayAverage', 'twoHundredDayAverage', 'profitMargins', 'sharesPercentSharesOut', 'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio', 'shortPercentOfFloat', 'earningsQuarterlyGrowth', 'trailingEps', 'forwardEps', '52WeekChange', 'SandP52WeekChange', 'lastDividendValue', 'currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice', 'recommendationMean']
        self.account_data_columns = ['timestamp', 'currentCash', 'currentStock', 'closingPrice', 'accountValue', 'buys', 'sells'],
        self.quote_table = "test_quote_data"
        self.account_table = "test_account_data"

def get_test_config()->BaseConfig:
    return BaseConfig()

def get_prod_config()->BaseConfig:
    new_config = BaseConfig()
    new_config.quote_table = "quote_data"
    new_config.account_table = "account_data"
    return new_config