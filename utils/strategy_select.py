from strategies.sma import moving_average_crossover
from strategies.rsi import rsi_indicator
from strategies.macd import macd_crossover
from pandas import DataFrame

def strategy_select(coin=DataFrame):
    while True:
        print("Choose a trading strategy:")
        print("1: SMA Crossover")
        print("2: RSI Threshold")
        print("3: MACD Crossover")
        
        choice = input("\nEnter strategy number 1-3: ")

        if choice == '1':
            result = moving_average_crossover(coin)
            print(result)
            continue
        elif choice == '2':
            result = rsi_indicator(coin)
            print(result)
            continue
        elif choice == '3':
            result = macd_crossover(coin)
            print(result)
            continue
        else:
            print("Invalid choice. Please enter a number 1-3")
            continue