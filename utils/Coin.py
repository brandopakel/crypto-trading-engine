import pandas as pd
from utils.coinbase_api_client import fetch_all_candles, aggregate_all_candles
from utils.strategy_select import strategy_select
from utils.plot import plot_strategy

class Coin:
    def __init__(self, product_id, candles=None):
        self.product_id = product_id
        self.df = pd.DataFrame(candles) if candles else pd.DataFrame()
    
    def fetch_candles(self):
        candles = fetch_all_candles(self.product_id)
        self.df = aggregate_all_candles(candles)

    def strategy_select(self):
        strategy = strategy_select(self.df)
        return strategy
    
    def plot(self, title ="Strategy Visualization", overlays=None, indicators=None, signals=""):
        plot_strategy(self.df, title, overlays, indicators)
