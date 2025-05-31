from pandas import DataFrame
from utils.plot import plot_strategy

def rsi_indicator(coin=DataFrame):
    coin = coin.copy()
    coin['close'] = coin['close'].astype(float)
    delta = coin['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain/avg_loss
    coin['RSI'] = 100 - (100/(1+rs))
    coin['RSI'] = coin['RSI'].astype(float)
    coin['signal'] = 0
    coin.loc[coin['RSI']<30, 'signal'] = 1
    coin.loc[coin['RSI']>70, 'signal'] = -1 

    coin.dropna(inplace=True)

    plot_strategy(coin, title="RSI Indicator", indicators=["RSI"], signal_col='signal')

    return coin