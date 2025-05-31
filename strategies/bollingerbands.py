from pandas import DataFrame
from utils.plot import plot_strategy

def defined_bollinger_bands_strategy(coin=DataFrame,window=20,num_std=2):
    coin = coin.copy()
    coin['close'] = coin['close'].astype(float)

    coin['MA'] = coin['close'].rolling(window).mean()
    coin['STD'] = coin['close'].rolling(window).std()

    coin['UpperBand'] = coin['MA'] + (num_std * coin['STD'])
    coin['LowerBand'] = coin['MA'] - (num_std * coin['STD'])

    coin['signal'] = 0
    coin.loc[coin['close'] < coin['LowerBand'], 'signal'] = 1
    coin.loc[coin['close'] > coin['UpperBand'], 'signal'] = -1

    coin.dropna(inplace=True)

    plot_strategy(coin, title="Bollinger Bands", overlays=["UpperBand", "LowerBand", "MA"], indicators=["STD"], signal_col='signal')

    return coin