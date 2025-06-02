from typing import List, Optional
import pandas as pd
from utils.Strategy import Strategy
from utils.plot import plot_strategy
from utils.logger import save_log

class MultiStrategyManager:
    def __init__(self, strategies: list[Strategy]):
        self.strategies = strategies
        self.signal_cols = []
        self.fib_levels = {}

    def apply_strategies(self, coin: pd.DataFrame):
        coin = coin.copy()
        self.signal_cols = []

        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__.replace("Strategy", "").lower()
            signal_col_name = f"{strategy_name}_signal"

            result = strategy.apply(coin)

            if 'signal' in result.columns:
                result = result.rename(columns={'signal': signal_col_name})
                #For testing : print(f"Renamed 'signal' to {signal_col_name}")
                if signal_col_name not in self.signal_cols:
                    self.signal_cols.append(signal_col_name)

            #Testing: new_cols = result.columns.difference(coin.columns)
            #Testing: coin[new_cols] = result[new_cols]

            for col in result.columns:
                if col not in coin.columns or not coin[col].equals(result[col]):
                    coin[col] = result[col]

            if hasattr(strategy, 'fib_levels'):
                self.fib_levels.update(strategy.fib_levels)
        
        coin.dropna(inplace=True)

        #Testing: save_log(coin, "/Users/bp/Documents/py_trading_rec/data/raw", "strategy_log")

        return coin
    
    def collect_plot_metadata(self, coin: pd.DataFrame) -> tuple:
        overlays = []
        indicators = []

        for strategy in self.strategies:
            overlays.extend(getattr(strategy, 'overlay_cols', []))
            indicators.extend(getattr(strategy, 'indicator_cols', []))
            #For testing : signal_cols.extend(getattr(strategy, 'signal_cols', []))
        
        return overlays, indicators, self.signal_cols
    
    def plot_combined(self, coin: pd.DataFrame, title: str="Multi-Strategy Overlay"):
        overlays, indicators, signal_cols = self.collect_plot_metadata(coin)

        #For testing: signal_col = signal_cols[0] if signal_cols else 'signal'
        if signal_cols:
            #For testing: print(signal_cols)
            #For testing: print(coin['fibonacciretracement_signal'].value_counts())
            #Test : signal_cols = [col for col in coin.columns if col.endswith('_signal')]
            #For testing: print(signal_cols)
            #For testing: print(coin.columns)
            valid_signal_cols = [col for col in signal_cols if col in coin.columns]

            if not valid_signal_cols:
                print("[Warning] No valid signal columns found in dataframe.")
            else:
                print("Signal columns used for aggregation:", valid_signal_cols)

            coin['signal'] = coin[valid_signal_cols].fillna(0).sum(axis=1).clip(-1,1)

            save_log(coin, "/Users/bp/Documents/py_trading_rec/data/raw", "strategy_log")
            
            signal_col = 'signal'
        else:
            signal_col = 'signal'

        return plot_strategy(
            coin,
            title=title,
            overlays=list(set(overlays)),
            indicators=list(set(indicators)),
            fib_levels=self.fib_levels,
            signal_col= signal_col
        )