def validate_granularity(start_t, end_t, user_input=str):
    granularity_map = {
        "ONE_MINUTE": 60,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 900,
        "THIRTY_MINUTE": 1800,
        "ONE_HOUR": 3600,
        "SIX_HOUR": 21600,
        "ONE_DAY": 86400
    }

    if user_input not in granularity_map:
        raise ValueError(f"Invalid granularity option: {user_input}. Choose from: {', '.join(granularity_map.keys())}")

    user_granularity_seconds = granularity_map[user_input]
    num_candles = (end_t - start_t) / user_granularity_seconds

    if num_candles > 350:
        raise ValueError(
            f"Time range too large for selected granularity '{user_input}': "
            f"{int(num_candles)} candles requested (max 350). "
            f"Reduce the time range or choose a coarser granularity."
        )

    return user_input