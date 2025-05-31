import pandas as pd
import numpy as np
from utils.plot import plot_strategy

def zscore_mean_reversion(coin: pd.DataFrame, window: int = 20, threshold: float = 1.5):
    coin = coin.copy()
    coin['close'] = coin['close'].astype(float)

    coin['MA'] = coin['close'].rolling(window).mean()
    coin['STD'] = coin['close'].rolling(window).std()
    coin['Z_Score'] = (coin['close']-coin['MA']) / coin['STD']

    coin['signal'] = 0
    coin.loc[coin['Z_Score'] < -threshold, 'signal'] = 1
    coin.loc[coin['Z_Score'] > threshold, 'signal'] = -1

    coin.dropna(inplace=True)

    plot_strategy(coin, title=f"Z-Score Mean Reversion Strategy (Window = {window}, Threshold = {threshold})", overlays=['MA'], indicators=['Z_Score'], signal_col='signal')

    return coin