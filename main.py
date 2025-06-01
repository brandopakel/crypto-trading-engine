from utils.helper import initial_recommendations, coin_selector, strategy_selector
from utils.Coin import Coin
from utils.Strategy import Strategy
from utils.product_id_finder import product_id
from utils.class_strategy_select import strategy_select
from utils.class_multi_strategy_select import multi_strategy_select
from utils.MultiStrategyManager import MultiStrategyManager
from utils.plot import plot_strategy
import pandas as pd


def main():
    initial_recommendations()

    id = product_id()
    coin = Coin(id)
    coin.get_candles()
    coin_candles = coin.fetch_candles()
    while coin_candles is not None:
        strategies = multi_strategy_select()
        manager = MultiStrategyManager(strategies)
        plottable_coin = manager.apply_strategies(coin_candles)
        print(plottable_coin)
        manager.collect_plot_metadata(plottable_coin)
        plot = manager.plot_combined(plottable_coin)
        choice = input("\n\nDo you want to strategize again? (Y/N): ")
        if choice.strip().lower() == 'y':
            continue
        elif choice.strip().lower() == 'n':
            print("\nThank you for your time.\n")
            break

if __name__ == "__main__":
    main()


""""
initial_recommendations()
coin = coin_selector()
while coin is not None:
    strategy_selector(coin)
    choice = input("\n\nDo you want to choose another strategy? (Y/N): ")
    if choice.strip().lower() == 'y':
        continue
    elif choice.strip().lower() == 'n':
        print("\nThank you for your time.\n")
        break
        """
