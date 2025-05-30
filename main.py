from utils.helper import initial_recommendations, coin_selector, strategy_selector


initial_recommendations()
coin = coin_selector()
while coin is not None:
    strategy_selector(coin)
    choice = input("\n\nDo you want to choose another strategy? (Y/N): ")
    if choice.strip().lower() == 'y':
        continue
    elif choice.strip().lower() == 'n':
        print("\nThank you for your time.")
        break