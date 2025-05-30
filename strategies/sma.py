from pandas import DataFrame

def moving_average_crossover(coin=DataFrame):
    coin['close'] = coin['close'].astype(float)
    coin['SMA_Short'] =coin['close'].rolling(5).mean()
    coin['SMA_Long'] = coin['close'].rolling(20).mean()

    coin['signal'] = 0
    coin.loc[coin['SMA_Short']>coin['SMA_Long'],'signal'] = 1
    coin.loc[coin['SMA_Short']<coin['SMA_Long'],'signal'] = -1

    coin.dropna(inplace=True)
    
    return coin