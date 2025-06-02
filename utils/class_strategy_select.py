from utils.Strategy import Strategy, BollingerBandsStrategy, MACDCrossoverStrategy,  RateOfChangeStrategy, RSIStrategy, MovingAverageCrossoverStrategy, ZScoreMeanReversionStrategy, FibonacciRetracementStrategy
from pandas import DataFrame
from utils.user_input import get_user_roc_inputs

def strategy_select(coin = DataFrame) -> Strategy:
    while True:
        strategy_map = {
            "bollinger": {
                "class": BollingerBandsStrategy,
                "desc": "Bollinger Bands (mean reversion using standard deviation)"
            },
            "macd": {
                "class": MACDCrossoverStrategy,
                "desc": "MACD Crossover (momentum-based crossover)"
            },
            "roc": {
                "class": RateOfChangeStrategy,
                "desc": "Rate of Change (momentum threshold strategy)"
            },
            "rsi": {
                "class": RSIStrategy,
                "desc": "RSI (Relative Strength Index based buy/sell)"
            },
            "sma": {
                "class": MovingAverageCrossoverStrategy,
                "desc": "Simple Moving Average crossover (trend following)"
            },
            "z": {
                "class": ZScoreMeanReversionStrategy,
                "desc": "Z-Score of Price vs Moving Average (mean reversion)"
            },
            "fibonacci": {
                "class": FibonacciRetracementStrategy,
                "desc": "Fibonacci Retracement Levels"
            }
        }

        print("\n📊 Available Trading Strategies:")
        for key, val in strategy_map.items():
            print(f"  - {key}: {val['desc']}")
        
        choice = input("\n🔍 Enter the name of a strategy to apply: ").strip().lower()

        if choice not in strategy_map:
            print("\n❌ Invalid choice. Please try again.")
            continue

        strategy_class = strategy_map.get(choice)['class']

        if not strategy_class:
            print("Invalid strategy choice. Try again.")
            continue

        #strategy_class = strategy_entry['class']

        if strategy_class == RateOfChangeStrategy:
            period, threshold = get_user_roc_inputs()
            return strategy_class(period, threshold)
        
        return strategy_class()