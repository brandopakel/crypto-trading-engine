from utils.user_input import get_user_selected_basename,get_user_selected_basesymbol,get_user_selected_product_id,set_user_granularity,set_user_time_interval
from utils.coinbase_api_client import get_coin_from_user_submitted_baseid, get_coin_from_user_submitted_basename,fetch_all_candles,get_product_id_from_basename,get_product_id_from_baseid, fetch_all_candles, aggregate_all_candles

def select_coin():
    while True:
        print("Coin Selection Menu: ")
        print("1 - Coin Name")
        print("2 - Coin Symbol")
        print("3 - Currency Pair (If you know what currency you were interested in trading with.)(e.g. 'BTC-USD')")
        choice = input("\nEnter your choice (1-3): ")
        if choice == "1":
            coin = get_coin_from_user_submitted_basename()
            basename = coin['Base-Name'].iloc[0]
            product_id = get_product_id_from_basename(basename)
            candles = fetch_all_candles(product_id)
            coin_nv = aggregate_all_candles(candles)
            print(f"\n{basename} Historical Data: ")
            return coin_nv
        elif choice == "2":
            coin = get_coin_from_user_submitted_baseid()
            baseid = coin['Base Currency ID'].iloc[0].upper()
            product_id = get_product_id_from_baseid(baseid)
            candles = fetch_all_candles(product_id)
            coin_nv = aggregate_all_candles(candles)
            print(f"\n{baseid} Historical Data: ")
            return coin_nv
        elif choice == "3":
            currency_pair = get_user_selected_product_id()
            base_id = currency_pair[:currency_pair.rfind("-")]
            candles = fetch_all_candles(currency_pair)
            coin_nv = aggregate_all_candles(candles)
            print(f"\n{base_id} Historical Data: ")
            return coin_nv
        else:
            print("Invalid choice. Please try again.")
            continue