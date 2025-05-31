from pandas import DataFrame
from utils.plot import plot_strategy

def rate_of_change_strategy(coin=DataFrame, period: int = 12, roc_threshold: float = 0):
    coin = coin.copy()
    coin['close'] = coin['close'].astype(float)

    coin['ROC'] = ((coin['close'] - coin['close'].shift(period)) / coin['close'].shift(period)) * 100

    coin['signal'] = 0
    coin.loc[coin['ROC'] > roc_threshold, 'signal'] = 1
    coin.loc[coin['ROC'] < roc_threshold, 'signal'] = -1

    coin.dropna(inplace=True)

    plot_strategy(coin, title="Rate of Change Strategy", indicators=["ROC"], signal_col='signal')

    return coin