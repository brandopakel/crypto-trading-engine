from strategies.rec import greatest_price_percentage_change_24h, greatest_volume_24h
from utils.coin_select import select_coin
from utils.strategy_select import strategy_select
from pandas import DataFrame

def initial_recommendations():
    print(greatest_price_percentage_change_24h())
    print("\n")
    print(greatest_volume_24h())
    print("\n")

def coin_selector():
    coin = select_coin()
    print(coin)
    return coin

def strategy_selector(coin=DataFrame):
    result = strategy_select(coin)
    print("\n")
    print(result)

def get_user_roc_inputs():
    while True:
        try:
            period = int(input("\nEnter the lookback period for ROC (e.g. 12): "))
            threshold = float(input("\nEnter the ROC threshold (e.g., 0): "))
            return period, threshold
        except ValueError:
            print("Invalid input. Please enter numeric values")