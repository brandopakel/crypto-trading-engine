import pandas as pd
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from typing import Optional
from plotly.subplots import make_subplots
from utils.plot import plot_strategy

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

        coin.dropna(inplace=True)

        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", overlays: Optional[list]=None, indicators: Optional[list] = None, signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="Bollinger Bands", overlays=["UpperBand", "LowerBand", "MA"], indicators=["STD"], signal_col='signal')

    
class RateOfChangeStrategy(Strategy):
    def __init__(self, period: int, threshold: float):
        self.period = period
        self.threshold = threshold
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)

        coin['ROC'] = ((coin['close'] - coin['close'].shift(self.period)) / coin['close'].shift(self.period)) * 100

        coin['signal'] = 0
        coin.loc[coin['ROC'] > self.threshold, 'signal'] = 1
        coin.loc[coin['ROC'] < self.threshold, 'signal'] = -1

        coin.dropna(inplace=True)

        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", overlays: Optional[list]=None, indicators: Optional[list] = None, signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="Rate of Change Strategy", indicators=["ROC"], signal_col='signal')

class MACDCrossoverStrategy(Strategy):
    def __init__(self, short_ema=12, long_ema=26, signal_ema=9):
        self.short_ema = short_ema
        self.long_ema = long_ema
        self.signal_ema = signal_ema
    
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

        coin.dropna(inplace=True)

        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", overlays: Optional[list]=None, indicators: Optional[list] = None, signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="MACD Strategy", indicators=["MACD", "Signal_Line"], signal_col='signal')

class RSIStrategy(Strategy):
    def __init__(self, window = 14):
        self.window = window
    
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

        coin.dropna(inplace=True)

        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", overlays: Optional[list]=None, indicators: Optional[list] = None, signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="RSI Indicator", indicators=["RSI"], signal_col='signal')

class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, short_window = 5, long_window = 20):
        self.short_window = short_window
        self.long_window = long_window
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)
        coin['SMA_Short'] =coin['close'].rolling(self.short_window).mean()
        coin['SMA_Long'] = coin['close'].rolling(self.long_window).mean()

        coin['signal'] = 0
        coin.loc[coin['SMA_Short']>coin['SMA_Long'],'signal'] = 1
        coin.loc[coin['SMA_Short']<coin['SMA_Long'],'signal'] = -1

        coin.dropna(inplace=True)
        
        return coin
      
    def plot(self, coin : pd.DataFrame, title:str = "", overlays: Optional[list]=None, indicators: Optional[list] = None, signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title="MA Crossover Strategy", overlays=["SMA_Short", "SMA_Long"], signal_col='signal')

class ZScoreMeanReversionStrategy(Strategy):
    def __init__(self, window : int = 20, threshold : float = 1.5):
        self.window = window
        self.threshold = threshold
    
    def apply(self, coin: pd.DataFrame) -> pd.DataFrame:
        coin = coin.copy()
        coin['close'] = coin['close'].astype(float)

        coin['MA'] = coin['close'].rolling(self.window).mean()
        coin['STD'] = coin['close'].rolling(self.window).std()
        coin['Z_Score'] = (coin['close']-coin['MA']) / coin['STD']

        coin['signal'] = 0
        coin.loc[coin['Z_Score'] < -self.threshold, 'signal'] = 1
        coin.loc[coin['Z_Score'] > self.threshold, 'signal'] = -1

        coin.dropna(inplace=True)
        
        return coin
    
    def plot(self, coin : pd.DataFrame, title:str = "", overlays: Optional[list]=None, indicators: Optional[list] = None, signal_col: str = "signal") -> go.Figure:
        plot_strategy(coin, title=f"Z-Score Mean Reversion Strategy (Window = {self.window}, Threshold = {self.threshold})", overlays=['MA'], indicators=['Z_Score'], signal_col='signal')