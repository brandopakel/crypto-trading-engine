import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Optional
from utils.user_input import get_user_order_inputs
import math

def auto_tuned_order(coin: pd.DataFrame, min_order: int = 2, max_order: Optional[int] = None, multiplier: float = 0.02) -> int:
    sensitivity = get_user_order_inputs()

    if not sensitivity:
        sensitivity = multiplier

    length = len(coin)
    order = int(length * sensitivity)

    if max_order:
        order = min(order, max_order)
    
    return max(min_order, order)

def find_local_extrema(coin: pd.DataFrame, order : int) -> pd.DataFrame:
    #coin = coin.copy()
    coin['FlowMinMax'] = 0

    max_idx = argrelextrema(coin['close'].values, np.greater, int(order))[0]
    min_idx = argrelextrema(coin['close'].values, np.less, int(order))[0]

    coin.loc[coin.index[max_idx], 'FlowMinMax'] = 1
    coin.loc[coin.index[min_idx], 'FlowMinMax'] = -1

    return coin

def is_valid_wave(df: pd.DataFrame, idx_seq) -> bool:
    closes = df.loc[idx_seq, 'close'].values

    return(
        closes[1] > closes[0] and
        closes[2] > closes[0] and
        closes[3] > closes[2] and
        closes[4] > closes[2] and
        closes[5] > closes[4]
    )

def distance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist 

def fail(reason):
    print(f"❌ {reason}")


def is_elliot_wave(coin: pd.DataFrame, i0, i1, i2, i3, i4, i5, ia, ib, ic):
    close = coin['close']

    # 1. Basic min/max alternation checks
    if not (coin.at[i0, 'FlowMinMax'] == -1 and coin.at[i2, 'FlowMinMax'] == -1 and
            coin.at[i4, 'FlowMinMax'] == -1 and coin.at[ia, 'FlowMinMax'] == -1 and
            coin.at[ic, 'FlowMinMax'] == -1):
        return False

    if not (coin.at[i1, 'FlowMinMax'] == 1 and coin.at[i3, 'FlowMinMax'] == 1 and
            coin.at[i5, 'FlowMinMax'] == 1 and coin.at[ib, 'FlowMinMax'] == 1):
        return False
    
    print("valid wave: close price 5:")
    print(close.loc[i5])
    print("valid wave: close price a:")
    print(close.loc[ia])
    
    # check impulse wave
    isi5TheTop = close.loc[i5] > close.loc[i1] and close.loc[i5] > close.loc[i2] and \
                 close.loc[i5] > close.loc[i3] and close.loc[i5] > close.loc[i4]
    if not isi5TheTop:
        return False

    if not close.loc[i1] > close.loc[i0]:
        return False

    if not close.loc[i1] > close.loc[i2]:
        return False

    if not close.loc[i2] > close.loc[i0]:
        return False

    if not close.loc[i3] > close.loc[i2]:
        return False

    w1Len = distance(i0, close.loc[i0], i1, close.loc[i1])

    if not close.loc[i3] > close.loc[i4]:
        return False

    if not close.loc[i4] > close.loc[i2]:
        return False

    w3Len = distance(i2, close.loc[i2], i3, close.loc[i3])

    if not close.loc[i4] > close.loc[i1]:
        return False

    if not close.loc[i5] > close.loc[i4]:
        return False

    if not close.loc[i5] > close.loc[i3]:
        return False

    w5Len = distance(i4, close.loc[i4], i5, close.loc[i5])

    if w3Len < w1Len and w3Len < w5Len:
        return False

    # uptrend
    isi5TheTop = close.loc[i5] > close.loc[ia] and close.loc[i5] > close.loc[ib] and close.loc[i5] > close.loc[ic]
    if not isi5TheTop:
        return False

    if not close.loc[i5] > close.loc[ia]:
        return False

    if not close.loc[i5] > close.loc[ib]:
        return False

    if not close.loc[ib] > close.loc[ia]:
        return False

    if not close.loc[ia] > close.loc[ic]:
        return False

    if not close.loc[ib] > close.loc[ic]:
        return False

    return True


def line(wa, wb, x):
    x1 = wa[0]
    y1 = wa[1]
    x2 = wb[0]
    y2 = wb[1]
    y = ((y2-y1)/(x2-x1))*(x-x1) + y1
    return y


def elliottWaveLinearRegressionError(coin: pd.DataFrame, wave: list[int], print_col: str = 'close') -> float:
    diffquad = 0
    try:
        for i in range(1, len(wave)):
            wa = [wave[i - 1], coin.loc[wave[i - 1], print_col]]
            wb = [wave[i], coin.loc[wave[i], print_col]]

            # loop through every x (index) between the wave points
            for xindex in range(wave[i - 1], wave[i]):
                if xindex not in coin.index:
                    continue  # skip missing indices safely
                yindex = coin.loc[xindex, print_col]
                yline = line(wa, wb, xindex)
                diffquad += (yindex - yline) ** 2

        return math.sqrt(diffquad) / (wave[-1] - wave[0])
    except KeyError as e:
        print(f"[Wave Regression Error] Missing index during regression: {e}")
        return float('inf')



def ElliottWaveDiscovery(coin: pd.DataFrame, measure: str = 'close'):

    def minRange(coin: pd.DataFrame, start, end):
        def localFilter(i):
            return isMin(coin,i)
        return filter(localFilter, list(range(start,end)))

    def maxRange(coin: pd.DataFrame, start, end):
        def localFilter(i):
            return isMax(coin,i)
        return filter(localFilter, list(range(start,end)))


    waves = []
    for i0 in minRange(coin,0,len(coin)):
        for i1 in maxRange(coin,i0+1,len(coin)):
            for i2 in minRange(coin,i1+1,len(coin)):
                for i3 in maxRange(coin,i2+1,len(coin)):
                    for i4 in minRange(coin,i3+1,len(coin)):
                        for i5 in maxRange(coin,i4+1,len(coin)):

                            isi5TheTop = coin[measure].iat[i5] > coin[measure].iat[i1] and coin[measure].iat[i5] > coin[measure].iat[i2] and coin[measure].iat[i5] > coin[measure].iat[i3] and coin[measure].iat[i5] > coin[measure].iat[i4]  
                            if isi5TheTop:

                                for ia in minRange(coin,i5+1,len(coin)):
                                    for ib in maxRange(coin,ia+1,len(coin)):
                                        for ic in minRange(coin,ib+1,len(coin)):
                                            wave = isElliottWave(coin,measure, i0,i1,i2,i3,i4,i5,ia,ib,ic)
                                            if wave is None:
                                                continue
                                            if not wave in waves:
                                                waves.append(wave)
                                                print(wave)

    return waves
    
def isMin(df: pd.DataFrame, i):
    return df['FlowMinMax'].iat[i] == -1

def isMax(df: pd.DataFrame, i):
    return df['FlowMinMax'].iat[i] == 1 

def isElliottWave(df: pd.DataFrame, value, i0,i1,i2,i3,i4,i5,ia,ib,ic):
    result = None
    # print(".")

    if not isMin(df,i0) or not isMin(df,i2) or not isMin(df,i4) or not isMin(df,ia) or not isMin(df,ic):
        return result

    if not isMax(df,i1) or not isMax(df,i3) or not isMax(df,i5) or not isMax(df,ib):
        return result

    isi5TheTop = df[value].iat[i5] > df[value].iat[i1] and df[value].iat[i5] > df[value].iat[i2] and df[value].iat[i5] > df[value].iat[i3] and df[value].iat[i5] > df[value].iat[i4]  
    if not isi5TheTop:
        return result

    if not df[value].iat[i1] > df[value].iat[i0]:
        return result

    if not df[value].iat[i1] > df[value].iat[i2]:
        return result
    
    if not df[value].iat[i2] > df[value].iat[i0]:
        return result
       
    if not df[value].iat[i3] > df[value].iat[i2]:
        return result

    # w1Len = np.abs(df[value].iat[i1]-df[value].iat[i0])
    # w2Len = np.abs(df[value].iat[i1]-df[value].iat[i2])
    w1Len = distance(i0,df[value].iat[i0],i1,df[value].iat[i1])
    # w2Len = calculateDistance(i1,df[value].iat[i1],i2,df[value].iat[i2])
    # if not w2Len < 2*w1Len:
    #     return result

    if not df[value].iat[i2] > df[value].iat[i0]:
        return result

    # result = [i0,i1,i2,i3]

    if not df[value].iat[i3] > df[value].iat[i4]:
        return result

    if not df[value].iat[i4] > df[value].iat[i2]:
        return result

    # w3Len = np.abs(df[value].iat[i3]-df[value].iat[i2])
    w3Len = distance(i2,df[value].iat[i2],i3,df[value].iat[i3])
    # w4Len = np.abs(df[value].iat[i4]-df[value].iat[i3])

    if not df[value].iat[i4] > df[value].iat[i1]:
        return result

    # result = [i0,i1,i2,i3,i4]

    if not df[value].iat[i5] > df[value].iat[i4]:
        return result

    if not df[value].iat[i5] > df[value].iat[i3]:
        return result

    # w5Len = np.abs(df[value].iat[i5]-df[value].iat[i4])
    w5Len = distance(i4,df[value].iat[i4],i5,df[value].iat[i5])

    if (w3Len < w1Len and w3Len < w5Len):
        return result

    # uptrend
    result = [i0,i1,i2,i3,i4,i5]

    isi5TheTop = df[value].iat[i5] > df[value].iat[ia]  and df[value].iat[i5] > df[value].iat[ib]  and df[value].iat[i5] > df[value].iat[ic]
    if not isi5TheTop:
        return result

    if not df[value].iat[i5] > df[value].iat[ia]:
        return result
    
    # waLen = calculateDistance(i5,df[value].iat[i5],ia,df[value].iat[ia])
    # wcLen = calculateDistance(ib,df[value].iat[ib],ic,df[value].iat[ic])

    # if waLen > wcLen:
    #     return result

    # if not (df[value].iat[i3] >= df[value].iat[ia] and df[value].iat[ia] >= df[value].iat[i4]):
    #     return result

    if not df[value].iat[i5] > df[value].iat[ib]:
        return result

    if not df[value].iat[ib] > df[value].iat[ia]:
        return result

    if not df[value].iat[ia] > df[value].iat[ic]:
        return result

    if not df[value].iat[ib] > df[value].iat[ic]:
        return result

    # if not df[value].iat[ia] > df[value].iat[ic]:
    #     return result

    # uptrend and retracement
    result = [i0,i1,i2,i3,i4,i5,ia,ib,ic]

    return result 

def filterWaveSet(waves, min_len=6, max_len=6, extremes=True):

    result = []
    for w in waves:
        l = len(w)
        if min_len <= l and l <= max_len:
            result.append(w)
    
    if not extremes:
        return result

    # find the max
    max = 0 
    for w in result:
        if w[len(w)-1] >= max:
            max = w[len(w)-1]
        
    #  filter the max
    result2 = []
    for w in result:
        if w[len(w)-1] == max:
            result2.append(w)

    # find the min
    min = max
    for w in result2:
        if w[0] <= min:
            min = w[0]    

    #  filter the min
    result = []
    for w in result2:
        if w[0] == min:
            result.append(w)

    return result