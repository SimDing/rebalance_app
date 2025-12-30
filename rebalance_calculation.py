
def rebalance_buy_only(current_values, target_weights, total_buy_value):
    """
    Calculate the buy-only rebalance amounts for a portfolio.

    Parameters:
    current_values (dict): A dictionary with asset names as keys and their current values as values.
    target_weights (dict): A dictionary with asset names as keys and their target weights (as decimals) as values.
    total_buy_value (float): The total amount available to buy.

    Returns:
    dict: A dictionary with asset names as keys and the amount to buy as values.
    """
    target_values = calculate_target_values(current_values, target_weights, total_buy_value)
    while any(current_values.get(asset, 0) >= target_values[asset] for asset in target_weights):
        target_weights = {asset: weight for asset, weight in target_weights.items() if current_values.get(asset, 0) < target_values[asset]}
        target_values = calculate_target_values(current_values, target_weights, total_buy_value)

    buy_amounts = {asset: max(0, target_values[asset] - current_values.get(asset, 0)) for asset in target_weights}

    return buy_amounts

def calculate_target_values(current_values, target_weights, additional_value):
    """
    Calculate the target values for each asset in a portfolio based on target weights.

    Parameters:
    current_values (dict): A dictionary with asset names as keys and their current values as values.
    target_weights (dict): A dictionary with asset names as keys and their target weights (as decimals) as values.

    Returns:
    dict: A dictionary with asset names as keys and their target values as values.
    """
    target_weights = normalize_weights(target_weights)
    total_current_value = sum(current_values.get(asset, 0) for asset in target_weights)
    total_target_value = total_current_value + additional_value
    target_values = {asset: total_target_value * weight for asset, weight in target_weights.items()}
    return target_values

def normalize_weights(weights):
    """
    Normalize a dictionary of weights so that they sum to 1.

    Parameters:
    weights (dict): A dictionary with asset names as keys and their weights (as decimals) as values.

    Returns:
    dict: A dictionary with normalized weights.
    """
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero.")
    return {asset: weight / total_weight for asset, weight in weights.items()}

def value_preserving_round(buy_amounts, decimals=2):
    """
    Round buy amounts while preserving the total value.

    Parameters:
    buy_amounts (dict): A dictionary with asset names as keys and their buy amounts as values.
    decimals (int): The number of decimal places to round to.

    Returns:
    dict: A dictionary with rounded buy amounts.
    """
    total = sum(buy_amounts.values())
    max_factor = (total + len(buy_amounts) * 10 ** (-decimals)) / total
    min_factor = 1 / max_factor
    rounded_total = round(total, decimals)
    while min_factor < max_factor:
        factor = (min_factor + max_factor) / 2
        adjusted = _round_values({asset: amount * factor for asset, amount in buy_amounts.items()}, decimals)
        adjusted_total = round(sum(adjusted.values()), decimals)
        if adjusted_total < rounded_total:
            min_factor = factor
        elif adjusted_total > rounded_total:
            max_factor = factor
        else:
            return adjusted
    raise ValueError("Could not preserve total value with rounding.")


def _round_values(buy_amounts, decimals=2):
    """
    Round buy amounts to a specified number of decimal places.

    Parameters:
    buy_amounts (dict): A dictionary with asset names as keys and their buy amounts as values.
    decimals (int): The number of decimal places to round to.

    Returns:
    dict: A dictionary with rounded buy amounts.
    """
    return {asset: round(amount, decimals) for asset, amount in buy_amounts.items()}

def kl_divergence(reference, actual):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.

    Parameters:
    actual (dict): A dictionary representing the true distribution.
    reference (dict): A dictionary representing the reference distribution.

    Returns:
    float: The KL divergence D_KL(P || Q).
    """
    import math
    q = normalize_weights(reference)
    p = normalize_weights(actual)
    divergence = 0.0
    for key in p:
        if p[key] > 0:
            divergence += p[key] * math.log(p[key] / q.get(key, 1e-10))
    return divergence

if __name__ == "__main__":
    # Example usage
    current_values = {
        'AssetA': 5000,
        'AssetB': 3000,
        'AssetC': 2000
    }
    target_weights = {
        'AssetA': 0.1,
        'AssetB': 0.3,
        'AssetC': 0.3
    }
    total_buy_value = 2000

    buy_amounts = rebalance_buy_only(current_values, target_weights, total_buy_value)
    print("Buy amounts for rebalance:")
    for asset, amount in buy_amounts.items():
        print(f"{asset}: {amount:.2f}")
    
    nats_serve = kl_divergence(reference={'a': 0.5, 'b': 0.5}, actual={'a': 0.50, 'b': 0.55})
    print(f"KL Divergence: {nats_serve:.2E} nats")
    nats_warn = kl_divergence(reference={'a': 0.5, 'b': 0.5}, actual={'a': 0.50, 'b': 0.51})
    print(f"KL Divergence: {nats_warn} nats")