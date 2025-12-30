import asyncio
import json
import os

from pytr.account import login
from pytr.api import TradeRepublicApi

from api_wrapper import ApiWrapper
from rebalance_calculation import rebalance_buy_only, value_preserving_round, kl_divergence

NO_IMPROVEMENT_WARNING = "Warning: The proposed buy orders do not improve the portfolio's alignment with target weights. This may be due to very good current alignment or insufficient buy power."
ALLOCATION_WARNING = "Note: The proposed buy orders only partially improve alignment with target weights (KL-divergence exceeds warning threshold)."
SERVE_ALLOCATION_WARNING = "Warning: The proposed buy orders are insufficient to achieve a good alignment with target weights (KL-divergence exceeds threshold). This may be due to large market movements, changes in target weights, or insufficient buy power."

WARN_DIVERGENCE_THRESHOLD = float(os.getenv("WARN_DIVERGENCE_THRESHOLD", 4.9e-5)) # default: warn when a 50/50 allocation shifts to 50/51 even after rebalancing
SERVE_DIVERGENCE_THRESHOLD = float(os.getenv("SERVE_DIVERGENCE_THRESHOLD", 1.13e-3)) # default: warn when a 50/50 allocation shifts to 50/55 even after rebalancing

async def main() -> None:
    target_allocation = get_target_allocation()

    # Initialize the API client
    api: TradeRepublicApi = login(store_credentials=True)
    api_wrapper = ApiWrapper(api)

    try:
        print("Fetching portfolio...")
        portfolio = await api_wrapper.request(api.compact_portfolio())
        positions = portfolio['positions']
        
        print("Collecting asset and ticker data...")
        data_tasks = list(map(lambda pos: collect_instrument_data(pos, api, api_wrapper), positions))
        collected_data = await asyncio.gather(*data_tasks)
        data_by_id = {item['instrument_id']: item for item in collected_data}

        current_values = {}
        for item in collected_data:
            current_values[item['instrument_id']] = item['net_worth']

        print("Looking up existing savings plans...")
        current_plans = await get_savings_plans_by_instrument(api_wrapper, api)
        default_amount, default_interval = guess_amount_and_interval_from_plans(current_plans, target_allocation)

        overall_investment: int = ask_for_wanted_overall_buy_value(default_amount)

        buy_amounts: dict[str, int] = create_tr_buy_amounts(current_values, target_allocation, overall_investment)
        for instrument_id, amount in buy_amounts.items():
            instrument_name = data_by_id[instrument_id]['details']['name']
            print(f"Buy {amount}\t of {instrument_name}")
        
        end_values = {k: current_values.get(k, 0) + buy_amounts.get(k, 0) for k in target_allocation.keys()}
        warn_on_unusual_allocation_divergence(current_values, end_values, target_allocation)
        
        if ask_for_confirmation("Proceed with setting up / modifying savings plans?"):
            #print("Current Savings Plans:")
            #print(json.dumps(current_plans, indent=2))
            queued_actions = []
            for instrument_id, amount in buy_amounts.items():
                instrument_name = data_by_id[instrument_id]['details']['name']
                if current_plans.get(instrument_id, {}).get('interval') == default_interval:
                    plan = current_plans[instrument_id]
                    plan_id = plan['id']
                    queued_actions.append((
                        None,
                        f"Changing amount of existing savings plan for {instrument_name} to {amount}.",
                        lambda plan_id=plan_id, instrument_id=instrument_id, amount=amount, plan=plan:
                            api_wrapper.request(api.change_savings_plan(plan_id, instrument_id, amount, **extract_start_date_params(plan)))
                    ))
                elif instrument_id in current_plans:
                    plan = current_plans[instrument_id]
                    plan_id = plan['id']
                    queued_actions.append((
                        f"This will change the interval of the savings plan for {instrument_name} to {default_interval}!",
                        f"Changing interval of existing savings plan for {instrument_name} to '{default_interval}' and amount to {amount}.",
                        lambda plan_id=plan_id, instrument_id=instrument_id, amount=amount:
                            api_wrapper.request(api.change_savings_plan(plan_id, instrument_id, amount, **create_start_date_params_for_interval(default_interval)))
                    ))
                else:
                    queued_actions.append((
                        f"This will create a new savings plan for {instrument_name}!",
                        f"Creating new savings plan for {instrument_name} with amount {amount} and interval '{default_interval}'.",
                        lambda instrument_id=instrument_id, amount=amount:
                        api_wrapper.request(api.create_savings_plan(instrument_id, amount, **create_start_date_params_for_interval(default_interval)))
                    ))
            
            warnings = [warning for warning, _, _ in queued_actions if warning is not None]
            continue_execution = True
            if warnings:
                print("\n ".join(warnings))
                continue_execution: bool = ask_for_confirmation("Do you want to continue?")
            if continue_execution:
                responses = await asyncio.gather(*[action() for _, _, action in queued_actions])
                success = True
                for i, response in enumerate(responses):
                    if not is_successful_response(response):
                        if success:
                            print("There were errors during the process:")
                        _, description, _ = queued_actions[i]
                        print(f"Failed action: {description}")
                        print(json.dumps(response, indent=2))
                        success = False
                if success:
                    print("Success")
    finally:
        print("Closing API connection...")
        await api_wrapper.close()

def warn_on_unusual_allocation_divergence(current_allocation, allocation_after_rebalance, target_allocation):
    current_divergence = kl_divergence(reference=target_allocation, actual=current_allocation)
    divergence_after_rebalance = kl_divergence(reference=target_allocation, actual=allocation_after_rebalance)

    if divergence_after_rebalance >= current_divergence:
        print(NO_IMPROVEMENT_WARNING.format(new_kd=divergence_after_rebalance, old_kd=current_divergence))
    
    if divergence_after_rebalance > SERVE_DIVERGENCE_THRESHOLD:
        print(SERVE_ALLOCATION_WARNING.format(new_kd=divergence_after_rebalance, old_kd=current_divergence, threshold=SERVE_DIVERGENCE_THRESHOLD))
    elif divergence_after_rebalance > WARN_DIVERGENCE_THRESHOLD:
        print(ALLOCATION_WARNING.format(new_kd=divergence_after_rebalance, old_kd=current_divergence, threshold=WARN_DIVERGENCE_THRESHOLD))

def get_exchange_preference():
    exchange_string = os.getenv('EXCHANGE', 'XETR,XFRA,LSX')
    return [ex.strip() for ex in exchange_string.split(',') if ex.strip()]

def get_target_allocation():
    path = os.getenv('ISIN_WEIGHTS', 'weights.json')
    try:
        with open(path, 'r') as f:
            wanted_instruments = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Weights file not found at path: {path}")

    validation_error = "Weights file must contain a JSON object with string keys (isin) and numeric values (target proportions)."
    
    if not isinstance(wanted_instruments, dict):
        raise ValueError(validation_error)
    for k, v in wanted_instruments.items():
        if not isinstance(k, str) or not isinstance(v, (int, float)):
            raise ValueError(validation_error)
    
    return wanted_instruments

def is_successful_response(response):
    return response.get('status') == 'succeeded'

def ask_for_confirmation(prompt) -> bool:
    while True:
        answer: str = input(f"{prompt} (y/N): ").strip().lower()
        if answer in ('y', 'n', ''):
            return answer == 'y'

def create_tr_buy_amounts(current_values, wanted_instruments, total_buy_value):
    # Minimum buy amount per instrument is 1 to avoid zero-amount savings plans
    min_buy_value = 1
    # Virtually add min_buy_amount for each instrument's current value and remove it from total buy value,
    # so that rebalance calculation can be done normally
    current_values = { k: current_values.get(k, 0) + min_buy_value for k in wanted_instruments.keys() }
    total_buy_value = total_buy_value - min_buy_value * len(wanted_instruments)
    buy_amounts = rebalance_buy_only(current_values, wanted_instruments, total_buy_value)
    # Round the buy amounts to whole numbers (EUR)
    buy_amounts = value_preserving_round(buy_amounts, decimals=0)
    # Add the minimum buy amount to each instrument
    buy_amounts = { k: buy_amounts.get(k, 0) + min_buy_value for k in wanted_instruments.keys() }
    # Convert to integer for API compatibility
    return { k: round(v) for k, v in buy_amounts.items() }

def create_start_date_params_for_interval(interval):
    if interval == 'monthly':
        return day_of_month_start_date_params(2)
    else:
        raise ValueError(f"Unsupported interval: {interval}")

def ask_for_wanted_overall_buy_value(default_value):
    while True:
        answer: str = input(f"Enter the total amount you want to invest for rebalancing (in EUR) [default: {default_value}]: ").strip()
        if answer == '':
            return default_value
        try:
            value = int(answer)
            if value > 0:
                return value
            else:
                print("Please enter a positive value.")
        except ValueError:
            print("Invalid input. Please enter an integer value.")

def guess_amount_and_interval_from_plans(plans_by_instrument, wanted_instruments):
    relevant_plans = { iid: plan for iid, plan in plans_by_instrument.items() if iid in wanted_instruments }
    if not relevant_plans:
        return 100, 'monthly'
    total_amount: int = sum(plan['amount'] for plan in relevant_plans.values())
    interval_counts = {}
    for plan in relevant_plans.values():
        interval = plan['interval']
        interval_counts[interval] = interval_counts.get(interval, 0) + 1
    most_common_interval = max(interval_counts.items(), key=lambda x: x[1])[0]
    return total_amount, most_common_interval

def extract_start_date_params(plan):
    start_date_object = plan['startDate']
    return {
        'start_date': start_date_object['nextExecutionDate'],
        'start_date_type': start_date_object['type'],
        'start_date_value': start_date_object['value'],
        'interval': plan['interval'],
    }

def day_of_month_start_date_params(day):
    return {
        "start_date": next_month_start_date(day),
        "start_date_type": "day_of_month",
        "start_date_value": day,
        "interval": "monthly",
    }

def next_month_start_date(day) -> str:
    from datetime import datetime, timedelta
    today: datetime = datetime.today()
    next_month: datetime = today.replace(day=28) + timedelta(days=4)
    day_next_month: datetime = next_month.replace(day=day)
    return day_next_month.strftime('%Y-%m-%d')

def ask_if_should_rebalance() -> bool:
    while True:
        answer: str = input("Do you want to proceed with the rebalancing orders? (y/N): ").strip().lower()
        if answer in ('y', 'n', ''):
            return answer == 'y'
        
async def get_savings_plans_by_instrument(api_wrapper, api):
    plans = await api_wrapper.request(api.savings_plan_overview())
    plans_by_instrument = {}
    for plan in plans.get('savingsPlans', []):
        instrument_id = plan.get('instrumentId')
        if instrument_id:
            plans_by_instrument[instrument_id] = plan
    return plans_by_instrument

async def collect_instrument_data(position, api, api_wrapper):
    instrument_id = position['instrumentId']
    details = await fetch_instrument_details(api, api_wrapper, instrument_id)
    price_info = await estimate_price_from_ticker(api, api_wrapper, instrument_id, details['exchange'])
    net_worth = float(position['netSize']) * price_info['estimated_price']
    return {
        'instrument_id': instrument_id,
        'details': details,
        'estimated_price': price_info['estimated_price'],
        'net_worth': net_worth,
    }



def calculate_net_worth(complete_detail_item):
    net_size = complete_detail_item['net_size']
    estimated_price = complete_detail_item['estimated_price']
    return net_size * estimated_price

async def fetch_instrument_details(api, api_wrapper, instrument_id):
    details = await api_wrapper.request(api.instrument_details(instrument_id))
    available_exchanges = details.get('exchangeIds')
    exchange_preference = get_exchange_preference()
    preferred_exchange = next((ex for ex in exchange_preference if ex in available_exchanges), available_exchanges[0] if available_exchanges else None)
    if preferred_exchange:
        data_at_exchange_list = [x for x in details.get('exchanges', []) if x.get('slug') == preferred_exchange]
    if not data_at_exchange_list:
        data_at_exchange_list = details.get('exchanges', [])
    if not data_at_exchange_list:
        raise ValueError(f"No exchange data available for instrument {instrument_id}")
    data_at_exchange = data_at_exchange_list[0]
    #print(json.dumps(details, indent=2))
    #raise ValueError("Debug stop")
    return {
        'name': details.get('shortName'),
        'instrument_id': instrument_id,
        'exchange': data_at_exchange.get('slug'),
        'symbol': data_at_exchange.get('symbolAtExchange'),
        'currency': details.get('notionalCurrency'),
    }

async def estimate_price_from_ticker(api, api_wrapper, instrument_id, exchange):
    search_results = await api_wrapper.request(api.ticker(instrument_id, exchange))
    ask = search_results.get('ask', {}).get('price')
    bid = search_results.get('bid', {}).get('price')
    if ask is None and bid is None:
        raise ValueError(f"No ask or bid price available for instrument {instrument_id} at exchange {exchange}")
    if ask is None:
        ask = bid
    if bid is None:
        bid = ask
    return { 'estimated_price': (float(ask) + float(bid)) / 2, 'instrument_id': instrument_id, 'exchange': exchange }


if __name__ == "__main__":
    asyncio.run(main())