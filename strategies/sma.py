from pandas import DataFrame
from utils.plot import plot_strategy

def moving_average_crossover(coin=DataFrame):
    coin = coin.copy()
    coin['close'] = coin['close'].astype(float)
    coin['SMA_Short'] =coin['close'].rolling(5).mean()
    coin['SMA_Long'] = coin['close'].rolling(20).mean()

    coin['signal'] = 0
    coin.loc[coin['SMA_Short']>coin['SMA_Long'],'signal'] = 1
    coin.loc[coin['SMA_Short']<coin['SMA_Long'],'signal'] = -1

    coin.dropna(inplace=True)

    plot_strategy(coin, title="MA Crossover Strategy", overlays=["SMA_Short", "SMA_Long"], signal_col='signal')
    
    return coin