from json import dumps
from coinbase.rest import RESTClient
import pandas as pd
import csv
from utils.user_input import get_user_selected_basename, get_user_selected_basesymbol, get_user_selected_product_id, set_user_time_interval, set_user_granularity
import datetime
from utils.validate_granularity import validate_granularity
from dotenv import load_dotenv
import os

load_dotenv("/Users/bp/Documents/py_trading_rec/env/keys.env")
api_key = os.getenv("KEY")
api_secret = os.getenv("PRIVATE_KEY")

#client = RESTClient(key_file="/Users/bp/Documents/py_trading_rec/cdp_api_key.json")
client = RESTClient(api_key=api_key,api_secret=api_secret)

global_product_list = pd.DataFrame()

def get_unix_time():
    current_time = client.get_unix_time()
    return current_time.epoch_seconds

def get_client_accounts():
    print("Requesting account data...")
    try:
        response = client.get_accounts()
        print("Response received")
        accounts = response.accounts
        print(f"Number of accounts: {len(accounts)}")
        df = pd.DataFrame([
            {
                "Currency" : acct["currency"],
                "Name" : acct["name"],
                "Available Balance" : float(acct["available_balance"]["value"]),
                "Type" : acct["type"],
                "Created" : acct["created_at"]
            }
            for acct in accounts if acct is not None
        ])
        return df
    except Exception as e:
        print(f"[Error] Failed to get accounts: {e}")
        return []

def get_best_bid_ask(product_id: str):
    print("requesting spread")
    try:
        return client.get_best_bid_ask(product_ids=product_id)
    except Exception as e:
        print(f"[Error] Failed to get ticker for {product_id}: {e}")
        return None

def get_public_products():
    print("requesting public product data...")
    try:
        public_products = client.get_public_products()
        print("\nacquired product data list")
        products = public_products.products
        print(f"\nNumber of available products: {len(products)}")
        public_product_df = pd.DataFrame([
            {
                "Product-ID" : product["product_id"],
                "Base-Name" : product["base_name"],
                "Price" : product["price"],
                "Price Percentage Change 24h" : float(product["price_percentage_change_24h"]),
                "Volume 24h" : float(product["volume_24h"]),
                "Base Currency ID" : product["base_currency_id"],
                "Quote Currency ID" : product["quote_currency_id"],
            }
            for product in products if product is not None
        ])
        #df.to_csv("/Users/bp/Documents/py_trading_rec/data/raw/public_products.csv")
        #global_product_list = public_product_df
        return public_product_df
    except Exception as e:
        print(f"[Error] Failed to get products: {e}")
        return None

def print_available_usdproductids():
    print("Available USD trading pairs: ")
    public_products = client.get_public_products()
    products = public_products.products
    usd_products = [p["product_id"] for p in products if p["quote_currency_id"] == "USD"]

    for pusd in sorted(usd_products):
        print(f" - {pusd}")

def get_coin_from_user_submitted_basename():
    baseid = get_user_selected_basename()
    public_products = client.get_public_products()
    products = public_products.products
    matched_df = pd.DataFrame([
        {
            "Product-ID" : match["product_id"],
            "Base-Name" : match["base_name"],
            "Price" : match["price"],
            "Price Percentage Change 24h" : float(match["price_percentage_change_24h"]),
            "Base Currency ID" : match["base_currency_id"],
            "Quote Currency ID" : match["quote_currency_id"],
        }
        for match in products if match is not None
    ])
    matched_coin_df = matched_df[matched_df["Base-Name"] == baseid]
    return matched_coin_df

def get_coin_from_user_submitted_baseid():
    baseid = get_user_selected_basesymbol()
    public_products = client.get_public_products()
    products = public_products.products
    matched_df = pd.DataFrame([
        {
            "Product-ID" : match["product_id"],
            "Base-Name" : match["base_name"],
            "Price" : match["price"],
            "Price Percentage Change 24h" : float(match["price_percentage_change_24h"]),
            "Base Currency ID" : match["base_currency_id"],
            "Quote Currency ID" : match["quote_currency_id"],
        }
        for match in products if match is not None
    ])
    matched_coin_df = matched_df[matched_df["Base Currency ID"] == baseid]
    return matched_coin_df

def get_public_market_trades():
    product_id = get_user_selected_product_id()
    trades = client.get_public_market_trades(product_id=product_id, limit=2)
    trade = trades.trades
    return trade

def get_market_trades():
    product_id = get_user_selected_product_id()
    trades = client.get_market_trades(product_id=product_id, limit=2)
    trade = trades.trades
    return trade

def get_product_candles(id=str):
    #product_id = get_user_selected_product_id()
    product_id = id
    start_dt, end_dt = set_user_time_interval()
    granularity = set_user_granularity()
    validated_granularity = validate_granularity(start_dt,end_dt,granularity)
    #print(start_dt)
    #print(end_dt)
    #current_time = get_unix_time()
    candles = client.get_candles(product_id=product_id, start=start_dt,end=end_dt,granularity=validated_granularity)
    #candle = candles.candles
    candle_dicts = [vars(candle) for candle in candles.candles]
    #print(type(candle_dicts))
    #print(type(candle))
    #print(type(candles))
    #print(type(candle))
    #print(candle[:2])
    #print(type(candle[0]))
    #print(candle[0])
    df = pd.DataFrame(candle_dicts)
    df['start'] = pd.to_datetime(df['start'].astype(int),unit='s')
    df = df.sort_values('start').reset_index(drop=True)
    return df
    #return candle

def get_product_id_from_basename(baseid: str) -> str | None:
    products = client.get_public_products().products
    """for p in products:
        if p.base_name.lower() == baseid.lower() and p.quote_currency_id == "USD":
            return p.product_id
    return None"""
    matching_products = [
        prod for prod in products if prod.base_name.lower() == baseid.lower()
    ]

    if not matching_products:
        print(f"[Error] No trading pairs found for base asset '{baseid}'.")
        return None
    
    print(f"\nAvailable quote currencies for {baseid.title()}:")
    quote_map = {}
    for i,product in enumerate(matching_products):
        print(f"{i+1} - {product.quote_currency_id}")
        quote_map[i+1] = product
    
    try:
        choice = int(input("\nEnter the number of the quote currency you want: "))
        selected = quote_map.get(choice)
        if selected:
            print(f"\n✅ Selected trading pair: {selected.product_id}")
            return selected.product_id
        else:
            print("[Error] Invalid selection.")
            return None
    except ValueError:
        print("[Error] Please enter a valid number.")
        return None
    
def get_product_id_from_baseid(baseid=str) -> str | None:
    products = client.get_public_products().products

    matching_products = [
        prod for prod in products if prod.base_currency_id.lower() == baseid.lower()
    ]

    if not matching_products:
        print(f"[Error] No trading pairs found for base asset '{baseid}'.")
        return None
    
    print(f"\nAvailable quote currencies for {baseid.upper()}:")
    quote_map = {}
    for i,product in enumerate(matching_products):
        print(f"{i+1} - {product.quote_currency_id}")
        quote_map[i+1] = product
    
    try:
        choice = int(input("\nEnter the number of the quote currency you want: "))
        selected = quote_map.get(choice)
        if selected:
            print(f"\n✅ Selected trading pair: {selected.product_id}")
            return selected.product_id
        else:
            print("[Error] Invalid selection.")
            return None
    except ValueError:
        print("[Error] Please enter a valid number.")
        return None