from strategies.sma import moving_average_crossover
from strategies.rsi import rsi_indicator
from strategies.macd import macd_crossover
from strategies.bollingerbands import defined_bollinger_bands_strategy
from strategies.roc import rate_of_change_strategy
from strategies.zscore import zscore_mean_reversion
from pandas import DataFrame

def strategy_select(coin=DataFrame):
    while True:
        print("\nChoose a trading strategy:")
        print("1: SMA Crossover")
        print("2: RSI Threshold")
        print("3: MACD Crossover")
        print("4: Bollinger Band Strategy")
        print("5: ROC Strategy")
        print("6: Z-Score Mean Reversion Strategy")
        
        choice = input("\nEnter strategy number 1-6: ")

        if choice == '1':
            result = moving_average_crossover(coin)
            return result
        elif choice == '2':
            result = rsi_indicator(coin)
            return result
        elif choice == '3':
            result = macd_crossover(coin)
            return result
        elif choice == '4':
            result = defined_bollinger_bands_strategy(coin)
            return result
        elif choice == '5':
            period = int(input("\nEnter the lookback period for ROC (e.g. 12): "))
            threshold = float(input("\nEnter the ROC threshold (e.g., 0): "))
            result = rate_of_change_strategy(coin,period,threshold)
            return result
        elif choice == '6':
            result = zscore_mean_reversion(coin)
            return result 
        else:
            print("Invalid choice. Please enter a number 1-6")
            continue