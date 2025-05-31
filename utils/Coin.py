import pandas as pd
from utils.coinbase_api_client import fetch_all_candles, aggregate_all_candles
from utils.strategy_select import strategy_select
from utils.plot import plot_strategy

class Coin:
    def __init__(self, product_id, candles=None):
        self.product_id = product_id
        self.df = pd.DataFrame(candles) if candles else pd.DataFrame()
    
    def get_candles(self):
        candles = fetch_all_candles(self.product_id)
        if not candles:
            print("⚠️ No candles returned!")
        self.df = aggregate_all_candles(candles)
        print(f"\n{self.product_id} Historical Data: ")
        print(self.df)

    def fetch_candles(self):
        return self.df

    def strategy_plot(self):
        strategy_select(self.df)