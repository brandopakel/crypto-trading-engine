import pandas as pd
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from typing import Optional
from plotly.subplots import make_subplots
from utils.plot import plot_strategy
from strategies.elliot_wave import find_local_extrema, is_valid_wave, elliottWaveLinearRegressionError, distance, ElliottWaveDiscovery, is_elliot_wave
import numpy as np
import math

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    Each subclass must implement the apply method.
    """
    @abstractmethod
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        pass

class BollingerBandsStrategy(Strategy):
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        self.overlay_cols = []
        self.indicator_cols = []
        self.signal_cols = []
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)

        coin['MA'] = coin['close'].rolling(self.window).mean()
        coin['STD'] = coin['close'].rolling(self.window).std()

        coin['UpperBand'] = coin['MA'] + (self.num_std * coin['STD'])
        coin['LowerBand'] = coin['MA'] - (self.num_std * coin['STD'])

        coin['signal'] = 0
        coin.loc[coin['close'] < coin['LowerBand'], 'signal'] = 1
        coin.loc[coin['close'] > coin['UpperBand'], 'signal'] = -1

        self.overlay_cols = ['UpperBand', 'LowerBand', 'MA']
        self.indicator_cols = ['STD']
        self.signal_cols = ['signal']

        return coin
    
    def apply_strategies(self, coin: pd.DataFrame, strategies: list[Strategy]) -> pd.DataFrame:
        coin = coin.copy()

        for strategy in strategies:
            result = strategy.apply(coin)
            new_cols = [col for col in result.columns if col not in coin.columns]
            coin[new_cols] = result[new_cols]

        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="Bollinger Bands", overlays=getattr(self, 'overlay_cols', []), indicators=getattr(self, 'indicator_cols', []), signal_col=signal_col)
    
class RateOfChangeStrategy(Strategy):
    def __init__(self, period: int, threshold: float):
        self.period = period
        self.threshold = threshold
        self.overlay_cols = []
        self.indicator_cols = []
        self.signal_cols = []
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)

        coin['ROC'] = ((coin['close'] - coin['close'].shift(self.period)) / coin['close'].shift(self.period)) * 100

        coin['signal'] = 0
        coin.loc[coin['ROC'] > self.threshold, 'signal'] = 1
        coin.loc[coin['ROC'] < self.threshold, 'signal'] = -1

        self.indicator_cols = ['ROC']
        self.signal_cols = ['signal']

        return coin
    
    def apply_strategies(self, coin: pd.DataFrame, strategies: list[Strategy]) -> pd.DataFrame:
        coin = coin.copy()

        for strategy in strategies:
            result = strategy.apply(coin)
            new_cols = [col for col in result.columns if col not in coin.columns]
            coin[new_cols] = result[new_cols]

        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="Rate of Change Strategy", overlays=getattr(self, 'overlay_cols', []), indicators=getattr(self, 'indicator_cols', []), signal_col=signal_col)

class MACDCrossoverStrategy(Strategy):
    def __init__(self, short_ema=12, long_ema=26, signal_ema=9):
        self.short_ema = short_ema
        self.long_ema = long_ema
        self.signal_ema = signal_ema
        self.overlay_cols = []
        self.indicator_cols = []
        self.signal_cols = []
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)
        coin['EMA_Short'] = coin['close'].ewm(span=self.short_ema,adjust=False).mean()
        coin['EMA_Long'] = coin['close'].ewm(span=self.long_ema,adjust=False).mean()
        coin['MACD'] = coin['EMA_Short'] - coin['EMA_Long']
        coin['Signal_Line'] = coin['MACD'].ewm(span=self.signal_ema,adjust=False).mean()

        coin['signal'] = 0
        coin.loc[coin['MACD']>coin['Signal_Line'],'signal'] = 1
        coin.loc[coin['MACD']<coin['Signal_Line'],'signal'] = -1

        self.indicator_cols = ['MACD', 'Signal_Line']
        self.signal_cols = ['signal']

        return coin
    
    def apply_strategies(self, coin: pd.DataFrame, strategies: list[Strategy]) -> pd.DataFrame:
        coin = coin.copy()

        for strategy in strategies:
            result = strategy.apply(coin)
            new_cols = [col for col in result.columns if col not in coin.columns]
            coin[new_cols] = result[new_cols]

        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="MACD Strategy", overlays=getattr(self, 'overlay_cols', []), indicators=getattr(self, 'indicator_cols', []), signal_col=signal_col)

class RSIStrategy(Strategy):
    def __init__(self, window = 14):
        self.window = window
        self.overlay_cols = []
        self.indicator_cols = []
        self.signal_cols = []
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)
        delta = coin['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(self.window).mean()
        avg_loss = loss.rolling(self.window).mean()

        rs = avg_gain/avg_loss
        coin['RSI'] = 100 - (100/(1+rs))
        coin['RSI'] = coin['RSI'].astype(float)
        coin['signal'] = 0
        coin.loc[coin['RSI']<30, 'signal'] = 1
        coin.loc[coin['RSI']>70, 'signal'] = -1

        self.indicator_cols = ['RSI'] 
        self.signal_cols = ['signal']

        return coin
    
    def apply_strategies(self, coin: pd.DataFrame, strategies: list[Strategy]) -> pd.DataFrame:
        coin = coin.copy()

        for strategy in strategies:
            result = strategy.apply(coin)
            new_cols = [col for col in result.columns if col not in coin.columns]
            coin[new_cols] = result[new_cols]

        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="RSI Indicator", overlays=getattr(self, 'overlay_cols', []), indicators=getattr(self, 'indicator_cols', []), signal_col=signal_col)

class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, short_window = 5, long_window = 20):
        self.short_window = short_window
        self.long_window = long_window
        self.overlay_cols = []
        self.indicator_cols = []
        self.signal_cols = []
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)
        coin['SMA_Short'] =coin['close'].rolling(self.short_window).mean()
        coin['SMA_Long'] = coin['close'].rolling(self.long_window).mean()

        coin['signal'] = 0
        coin.loc[coin['SMA_Short']>coin['SMA_Long'],'signal'] = 1
        coin.loc[coin['SMA_Short']<coin['SMA_Long'],'signal'] = -1

        self.overlay_cols = ['SMA_Short', 'SMA_Long']
        self.signal_cols = ['signal']
        
        return coin
    
    def apply_strategies(self, coin: pd.DataFrame, strategies: list[Strategy]) -> pd.DataFrame:
        coin = coin.copy()

        for strategy in strategies:
            result = strategy.apply(coin)
            new_cols = [col for col in result.columns if col not in coin.columns]
            coin[new_cols] = result[new_cols]

        return coin
      
    def plot(self, coin : pd.DataFrame, title:str = "", signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="MA Crossover Strategy", overlays=getattr(self, 'overlay_cols', []), indicators=getattr(self, 'indicator_cols', []), signal_col=signal_col)

class ZScoreMeanReversionStrategy(Strategy):
    def __init__(self, window : int = 20, threshold : float = 1.5):
        self.window = window
        self.threshold = threshold
        self.overlay_cols = []
        self.indicator_cols = []
        self.signal_cols = []
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)

        coin['MA'] = coin['close'].rolling(self.window).mean()
        coin['STD'] = coin['close'].rolling(self.window).std()
        coin['Z_Score'] = (coin['close']-coin['MA']) / coin['STD']

        coin['signal'] = 0
        coin.loc[coin['Z_Score'] < -self.threshold, 'signal'] = 1
        coin.loc[coin['Z_Score'] > self.threshold, 'signal'] = -1

        self.overlay_cols = ['MA']
        self.indicator_cols = ['Z_Score']
        self.signal_cols = ['signal']
        
        return coin
    
    def apply_strategies(self, coin: pd.DataFrame, strategies: list[Strategy]) -> pd.DataFrame:
        coin = coin.copy()

        for strategy in strategies:
            result = strategy.apply(coin)
            new_cols = [col for col in result.columns if col not in coin.columns]
            coin[new_cols] = result[new_cols]

        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title=f"Z-Score Mean Reversion Strategy (Window = {self.window}, Threshold = {self.threshold})", overlays=getattr(self, 'overlay_cols', []), indicators=getattr(self, 'indicator_cols', []), signal_col=signal_col)

class FibonacciRetracementStrategy(Strategy):
    def __init__(self):
        self.overlay_cols = []
        self.indicator_cols = []
        self.signal_cols = []
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:        
        highest_high = coin['high'].astype(float).max()
        lowest_low = coin['low'].astype(float).min()

        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)
        coin['high'] = coin['high'].astype(float)
        coin['low'] = coin['low'].astype(float)
        coin['volume'] = coin['volume'].astype(float)

        if highest_high == lowest_low:
            print("Warning] High and Low are equal — skipping Fibonacci calculation.")
            coin['signal'] = 0
            return coin
        
        diff = highest_high - lowest_low

        self.fib_levels = {
            "Fib_0.0": highest_high,
            "Fib_0.236": highest_high - (0.236*diff),
            "Fib_0.382": highest_high - (0.382*diff),
            "Fib_0.50": highest_high - (0.50*diff),
            "Fib_0.618": highest_high - (0.618*diff),
            "Fib_0.786": highest_high - (0.786*diff),
            "Fib_1.0": lowest_low
        }

        coin['signal'] = 0

        for label, level in self.fib_levels.items():
            coin.loc[
                (coin['close'] > level) & (coin['close'].shift(1) < level) & (coin['volume'] > coin['volume'].shift(1)), 'signal'
            ] = 1
            coin.loc[
                (coin['close'] < level) & (coin['close'].shift(1) > level) & (coin['volume'] > coin['volume'].shift(1)), 'signal'
            ] = -1

        self.signal_cols = ['signal']

        return coin
    
    def plot(self,coin:pd.DataFrame) -> go.Figure:
        return plot_strategy(coin, title="Fibonacci Retracement", fib_levels=self.fib_levels, signal_col='signal')

class ElliotWaveStrategy(Strategy):
    def __init__(self, order: int):
        self.order = order
        self.overlay_cols = []
        self.wave_labels = ['0','1','2','3','4','5','a','b','c']
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        # Detect extrema on a copy
        coin = find_local_extrema(coin, self.order)

        # Get only extrema points
        coin_extrema = coin[coin['FlowMinMax'] != 0].copy()
        extrema_points = coin_extrema.index.tolist()
        print(extrema_points)

        # Initialize columns in the full DataFrame
        for label in self.wave_labels:
            col_name = f'ew_{label}'
            if col_name not in coin.columns:
                coin[col_name] = float('nan')

        candidate_waves = []
        for i in range(len(extrema_points) - 8):
            wave = extrema_points[i:i + 9]
            print(coin.index.min(), coin.index.max())
            print(set(wave) - set(coin.index))
            print(f"Checked wave: {wave}")
            if is_elliot_wave(coin_extrema, *wave):
                print(f"Valid wave found at indices: {wave}")
                for label, idx in zip(self.wave_labels, wave):
                    if idx in coin_extrema.index:
                        price = coin_extrema.loc[idx, 'close']
                        print(f"Wave {label} at index {idx} has close price: {price}")
                    else:
                        print(f"Index {idx} not found in DataFrame")
                score = -elliottWaveLinearRegressionError(coin_extrema, wave, 'close')
                candidate_waves.append((wave, score))

        if not candidate_waves:
            print("No valid Elliott Waves found.")
            return coin

        selected_waves = []
        used_indices = set()
        for wave, score in sorted(candidate_waves, key=lambda x: x[1], reverse=True):
            if not any(idx in used_indices for idx in wave):
                selected_waves.append(wave)
                used_indices.update(wave)

        # Annotate into full coin DataFrame
        for wave in selected_waves:
            for i, idx in enumerate(wave):
                if i < len(self.wave_labels):
                    col_name = f'ew_{self.wave_labels[i]}'

                    if idx in coin.index:
                        coin.at[idx, col_name] = coin.at[idx, 'close']

                        if col_name not in self.overlay_cols:
                            self.overlay_cols.append(col_name)

        return coin