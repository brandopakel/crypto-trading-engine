from utils.helper import initial_recommendations, coin_selector, strategy_selector
from utils.Coin import Coin
from utils.Strategy import Strategy
from utils.product_id_finder import product_id
from utils.class_strategy_select import strategy_select
from utils.plot import plot_strategy


initial_recommendations()

#coin = coin_selector()

id = product_id()
coin = Coin(id)
coin.get_candles()
coin_candles = coin.fetch_candles()
strategy = strategy_select()
plottable_df = strategy.apply(coin_candles)
plot = strategy.plot(plottable_df)

""""
while coin is not None:
    strategy_selector(coin)
    choice = input("\n\nDo you want to choose another strategy? (Y/N): ")
    if choice.strip().lower() == 'y':
        continue
    elif choice.strip().lower() == 'n':
        print("\nThank you for your time.\n")
        break
        """
