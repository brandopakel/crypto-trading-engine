from typing import List, Optional
import pandas as pd
from utils.Strategy import Strategy
from utils.plot import plot_strategy
from utils.logger import save_log

class MultiStrategyManager:
    def __init__(self, strategies: list[Strategy]):
        self.strategies = strategies
        self.signal_cols = []

    def apply_strategies(self, coin: pd.DataFrame):
        coin = coin.copy()

        for strategy in self.strategies:
            result = strategy.apply(coin)

            strategy_name = strategy.__class__.__name__.replace("Strategy", "").lower()
            signal_col_name = f"{strategy_name}_signal"

            if 'signal' in result.columns:
                result = result.rename(columns={'signal': signal_col_name})
                #For testing : print(f"Renamed 'signal' to {signal_col_name}")
                self.signal_cols.append(signal_col_name)

            new_cols = result.columns.difference(coin.columns)
            coin[new_cols] = result[new_cols]
        
        coin.dropna(inplace=True)

        save_log(coin, "/Users/bp/Documents/py_trading_rec/data/raw", "strategy_log")

        return coin
    
    def collect_plot_metadata(self, coin: pd.DataFrame) -> tuple:
        overlays = []
        indicators = []

        for strategy in self.strategies:
            overlays.extend(getattr(strategy, 'overlay_cols', []))
            indicators.extend(getattr(strategy, 'indicator_cols', []))
            #For testing : signal_cols.extend(getattr(strategy, 'signal_cols', []))

            """strategy_name = strategy.__class__.__name__.replace("Strategy","").lower()
            signal_col_name = f"{strategy_name}_signal"
            if signal_col_name in coin.columns:
                signal_cols.append(signal_col_name)"""
        
        return overlays, indicators, self.signal_cols
    
    def plot_combined(self, coin: pd.DataFrame, title: str="Multi-Strategy Overlay"):
        overlays, indicators, signal_cols = self.collect_plot_metadata(coin)

        #For testing: signal_col = signal_cols[0] if signal_cols else 'signal'
        if signal_cols:
            signal_cols = [col for col in coin.columns if col.endswith('_signal')]
            #For testing: print(signal_cols)
            #For testing: print(coin.columns)
            coin['signal'] = coin[signal_cols].sum(axis=1).clip(-1,1)
            signal_col = 'signal'
        else:
            signal_col = 'signal'

        return plot_strategy(
            coin,
            title=title,
            overlays=list(set(overlays)),
            indicators=list(set(indicators)),
            signal_col=signal_col
        )