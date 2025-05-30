from utils.coinbase_api_client import get_public_products
from utils.logger import save_log

def greatest_price_percentage_change_24h():
    #global counter
    #counter+=1
    df = get_public_products()
    df_select = df[['Product-ID','Base-Name','Price','Price Percentage Change 24h']]
    sorted_df = df_select.sort_values("Price Percentage Change 24h", ascending=False)
    #sorted_df.to_csv(f"/Users/bp/Documents/py_trading_rec/data/logs/{counter}.csv")
    save_log(sorted_df,folder="/Users/bp/Documents/py_trading_rec/data/logs",base_name="percentage_change_24h_log")
    print("Showcase of the highest and lowest 24h percentage change for all available coins: \n")
    return sorted_df

def greatest_volume_24h():
    df = get_public_products()
    df_select = df[['Product-ID','Base-Name','Price','Volume 24h']]
    sorted_df = df_select.sort_values("Volume 24h", ascending=False)
    save_log(sorted_df,folder="/Users/bp/Documents/py_trading_rec/data/logs",base_name="volume_24h_log")
    print("Showcase of the greatest and least trading volume for all available coins: \n")
    return sorted_df