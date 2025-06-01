from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def get_user_selected_product_id():
    user_selected_product_id = input("\nEnter a coin currency pair (e.g. BTC-USD): ").strip().upper()
    return user_selected_product_id

def get_user_selected_basename():
    user_selected_basename = input("\nEnter your coin name: ").strip().upper()
    return user_selected_basename

def get_user_selected_basesymbol():
    user_selected_basesymbol = input("\nEnter your coin currency ID: ").strip().upper()
    return user_selected_basesymbol

def set_user_time_interval():
    print("\nEnter time interval (e.g. 1h, 6h, 1d, 7d, 1m, 3m)")
    interval = input(">> ").strip().lower()

    now = datetime.now()

    if interval.endswith("h"):
        hours = int(interval.rstrip("h"))
        start = now - timedelta(hours=hours)
    elif interval.endswith("d"):
        days = int(interval.rstrip("d"))
        start = now - timedelta(days=days)
    elif interval.endswith("m"):
        months = int(interval.rstrip("m"))
        start = now - relativedelta(months=months)
    else:
        print("❌ Invalid format. Use formats like '1h', '24h', '1d', '7d', '1m', '3m'.")
        return None, None

    start_unix = int(start.timestamp())
    end_unix = int(now.timestamp())

    return start_unix, end_unix

def set_user_granularity():
    options = {
        "1": "ONE_MINUTE",
        "2": "FIVE_MINUTE",
        "3": "FIFTEEN_MINUTE",
        "4": "THIRTY_MINUTE",
        "5": "ONE_HOUR",
        "6": "TWO_HOUR",
        "7": "SIX_HOUR",
        "8": "ONE_DAY"
    }

    print("\nSelect Granularity: ")
    
    for key, val in options.items():
        print(f"{key}: {val.replace('_', ' ').title()}")
    
    choice = input("\nEnter choice (1-8): ").strip()

    return options.get(choice,"ONE_HOUR")

def get_user_roc_inputs():
    while True:
        try:
            period = int(input("\nEnter the lookback period for ROC (e.g. 12): "))
            threshold = float(input("\nEnter the ROC threshold (e.g., 0): "))
            return period, threshold
        except ValueError:
            print("Invalid input. Please enter numeric values")