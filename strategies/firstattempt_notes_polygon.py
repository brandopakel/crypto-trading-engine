# python implemented trading engine
# Trading engine stack:
# Dashboard - Market Data / Statistical Information
# Securitization - Login and Authentication
# Order execution capabilities
# Conveniant CRM and admin/user functionality
# Fast searching and filtering tools
# Alerts - News Feed ---- ?? --> Nah (extra)
# Withdrawals and Payments 
# -- Portfolio Management (View transaction history, check portfolio balances, assess performance)

##### Server build: 
# Has to include requests both from front to backend to attain live market data
# This will be the backend server for *1 receiving requests and *2 processing requests
# where to find public market information....... yahoo ???? so general ughhh, is not fast, overcrowded with requests already... ; 

# please remember --no-warn-script-location flag to supress pip install

#from polygon import RESTClient
#client = RESTClient(api_key="moi6XQkockUQOIhMYNPl9f65mBXXHcZH")

#example ticker test -> NVDA
# ticker = 'NVDA'
#quote = client.get_last_quote(ticker=ticker)
#print(quote)

#import urllib.request
#contents = urllib.request.urlopen("https://api.polygon.io/v2/aggs/ticker/NVDA/prev?adjusted=true&apiKey=moi6XQkockUQOIhMYNPl9f65mBXXHcZH").read()
#print(contents)
#import requests
#print(requests.get(url="https://api.polygon.io/v2/aggs/ticker/NVDA/prev?adjusted=true&apiKey=moi6XQkockUQOIhMYNPl9f65mBXXHcZH"))

import requests
import json
r = requests.get("https://api.polygon.io/v3/reference/tickers/BTC?apiKey=moi6XQkockUQOIhMYNPl9f65mBXXHcZH")
r=r.json()
print(json.dumps(r, indent=3))

