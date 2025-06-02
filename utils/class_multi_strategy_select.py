from utils.Strategy import Strategy, BollingerBandsStrategy, MACDCrossoverStrategy,  RateOfChangeStrategy, RSIStrategy, MovingAverageCrossoverStrategy, ZScoreMeanReversionStrategy, FibonacciRetracementStrategy
from pandas import DataFrame
from utils.user_input import get_user_roc_inputs

def multi_strategy_select() -> list[Strategy]:
    selected_strategies = []

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

    while True:
        choice = input("\n🔍 Enter the name of a strategy to apply: (or type 'done' to finish): ").strip().lower()
        if choice == "done":
            break
        if choice in strategy_map:
            strategy_class = strategy_map.get(choice)['class']
            if strategy_class == RateOfChangeStrategy:
                period, threshold = get_user_roc_inputs()
                selected_strategies.append(strategy_class(period, threshold))
            else:
                selected_strategies.append(strategy_class())
        else:
            print("\n❌ Invalid choice. Please try again.")
    
    return selected_strategies

