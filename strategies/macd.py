from pandas import DataFrame

def macd_crossover(coin=DataFrame, short_ema=12, long_ema=26, signal_ema=9):
    coin = coin.copy()
    coin['close'] = coin['close'].astype(float)
    coin['EMA_Short'] = coin['close'].ewm(span=short_ema,adjust=False).mean()
    coin['EMA_Long'] = coin['close'].ewm(span=long_ema,adjust=False).mean()
    coin['MACD'] = coin['EMA_Short'] - coin['EMA_Long']
    coin['Signal_Line'] = coin['MACD'].ewm(span=signal_ema,adjust=False).mean()

    coin['signal'] = 0
    coin.loc[coin['MACD']>coin['Signal_Line'],'signal'] = 1
    coin.loc[coin['MACD']<coin['Signal_Line'],'signal'] = -1

    coin.dropna(inplace=True)

    return coin