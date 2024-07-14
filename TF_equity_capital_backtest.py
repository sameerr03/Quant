# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 22:52:48 2024

@author: SameerRangwala
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime as dt
import hf_functions as hf
import yfinance as yf
from hmmlearn import hmm
import sys
import os
import contextlib
import warnings
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import math

@contextlib.contextmanager
def suppress_output():
    # Determine the appropriate null device for the operating system
    if os.name == 'nt':
        null_device = 'nul'
    else:
        null_device = '/dev/null'
    
    with open(null_device, 'w') as fnull:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
def calculate_statistics(pnl_list):
    total_trades = len(pnl_list)
    total_profit = sum(pnl_list)
    wins = [pnl for pnl in pnl_list if pnl > 0]
    losses = [pnl for pnl in pnl_list if pnl <= 0]
    
    winrate = len(wins) / total_trades if total_trades > 0 else 0
    avg_profit = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    avg_pnl = total_profit / total_trades if total_trades > 0 else 0
    
    return {
        "total_trades": total_trades,
        "total_profit": total_profit,
        "winrate": winrate,
        "average_profit": avg_profit,
        "average_loss": avg_loss,
        "average_pnl": avg_pnl
    }

def print_statistics(stats, strategy_name, with_cost):
    cost_label = "with cost" if with_cost else "without cost"
    print(f"\nStatistics for {strategy_name} strategy ({cost_label}):")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Total Profit: {stats['total_profit']:.2f}")
    print(f"Winrate: {stats['winrate']:.2%}")
    print(f"Average Profit: {stats['average_profit']:.2f}")
    print(f"Average Loss: {stats['average_loss']:.2f}")
    print(f"Average P&L: {stats['average_pnl']:.2f}")
    print("-" * 40)

def calculate_volatility(returns):
    return np.std(returns)

def calculate_beta(strategy_returns, market_returns):
    strategy_returns = strategy_returns.reshape(-1, 1)
    market_returns = market_returns.reshape(-1, 1)
    reg = LinearRegression().fit(market_returns, strategy_returns)
    return reg.coef_[0][0]

def calculate_max_drawdown(cumpnl):
    roll_max = np.maximum.accumulate(cumpnl)
    drawdown = roll_max - cumpnl
    max_drawdown = np.max(drawdown)
    return max_drawdown

# Calculate and print Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    mean_return = np.mean(returns)
    return_std = np.std(returns)
    sharpe_ratio = (mean_return - risk_free_rate) / return_std
    return sharpe_ratio

# Calculate alpha using the regression intercept
def calculate_alpha(strategy_returns, market_returns, beta):
    market_returns = market_returns.reshape(-1, 1)
    reg = LinearRegression().fit(market_returns, strategy_returns)
    alpha = reg.intercept_[0]
    return alpha

def calculate_positions(df, min_holding_period=3, crossover_threshold=0):
    """
    Calculate trading positions based on MACD and Signal.

    Args:
        df (pd.DataFrame): The input dataframe with 'MACD' and 'Signal' columns.
        min_holding_period (int): The minimum number of days to hold a position before changing.
        crossover_threshold (float): The threshold for the MACD-Signal difference to confirm crossover.

    Returns:
        pd.Series: The positions (+1 for buy, -1 for sell).
    """
    positions = []
    current_position = 0
    holding_period = 0
    
    for i in range(len(df)):
        if holding_period < min_holding_period:
            positions.append(current_position)
            holding_period += 1
            continue
        
        macd_diff = df['MACD'].iloc[i] - df['Signal'].iloc[i]
        
        if macd_diff > crossover_threshold:
            if current_position != 1:
                current_position = 1
                holding_period = 0
        elif macd_diff < -crossover_threshold:
            if current_position != -1:
                current_position = -1
                holding_period = 0
        
        positions.append(current_position)
        holding_period += 1
    
    return pd.Series(positions, index=df.index)

def smooth_states_with_ema(states, span=3, threshold=0.5):
    """
    Smooth the state predictions using an exponential moving average (EMA).

    Args:
        states (pd.Series): The input series of state predictions.
        span (int): The span for the EMA. Default is 3.
        threshold (float): The threshold for converting the EMA to binary states. Default is 0.5.

    Returns:
        pd.Series: The smoothed binary state predictions.
    """
    # Calculate the EMA of the states
    ema_states = states.ewm(span=span, adjust=False).mean()
    
    # Convert the EMA to binary states based on the threshold
    smoothed_states = (ema_states >= threshold).astype(int)
    
    return smoothed_states

def og(og_df):
    # Example usage with the og_df dataframe
    og_df['Position'] = calculate_positions(og_df, min_holding_period=3, crossover_threshold=0)
    
    
    # Initialize the first value of sbp based on the initial conditions
    sbp = [0]
    
    # Iterate through the DataFrame to calculate sbp
    for i in range(1, len(og_df)):
        if og_df['Smoothed_State'].iloc[i] == 1 and og_df['Smoothed_State'].iloc[i - 1] == 1:
            sbp.append(og_df['Position'].iloc[i])
        elif og_df['Smoothed_State'].iloc[i] == 0 and og_df['Smoothed_State'].iloc[i - 1] == 0:
            sbp.append(0)
        elif og_df['Smoothed_State'].iloc[i] != og_df['Smoothed_State'].iloc[i - 1]:
            sbp.append(sbp[i - 1])
        else:
            sbp.append(0)
    
    # Add sbp to the DataFrame
    og_df['SBP'] = sbp
    
    
    # Initialize the Trade and Reason columns
    og_df['Trade'] = np.nan
    og_df['Reason'] = np.nan
    
    # Variables to track trade information
    entry_price = None
    highest_price = None
    lowest_price = None
    active_trade = None  # Can be 'long' or 'short'
    last_sbp = None
    
    for i in range(1, len(og_df)):
        current_price = og_df['Close'].iloc[i]
        current_sbp = og_df['SBP'].iloc[i]
        
        # Handle entry conditions
        if active_trade is None and current_sbp != last_sbp:
            if current_sbp == 1:
                entry_price = current_price
                highest_price = current_price
                lowest_price = None
                active_trade = 'long'
                og_df.at[i, 'Trade'] = 'Enter Long'
                last_sbp = current_sbp
            elif current_sbp == -1:
                entry_price = current_price
                lowest_price = current_price
                highest_price = None
                active_trade = 'short'
                og_df.at[i, 'Trade'] = 'Enter Short'
                last_sbp = current_sbp
        
        # Handle exit conditions for long trades
        elif active_trade == 'long':
            highest_price = max(highest_price, current_price)
    
            if (current_price < entry_price) and (current_price / entry_price - 1 <= -0.05):  # Stoploss
                og_df.at[i, 'Trade'] = 'Exit Long'
                og_df.at[i, 'Reason'] = 'Stoploss'
                active_trade = None
                entry_price = None
                highest_price = None
            elif (current_price > entry_price) and (current_price / entry_price - 1 >= 0.05):  # Takeprofit
                og_df.at[i, 'Trade'] = 'Exit Long'
                og_df.at[i, 'Reason'] = 'Takeprofit'
                active_trade = None
                entry_price = None
                highest_price = None
            elif (current_sbp == 0 and current_price > entry_price):  # Start trailing stoploss when sbp changes
                trailing_stoploss_price = highest_price * 0.99
                if current_price < trailing_stoploss_price:
                    og_df.at[i, 'Trade'] = 'Exit Long'
                    og_df.at[i, 'Reason'] = 'Trailing stoploss'
                    active_trade = None
                    entry_price = None
                    highest_price = None
                last_sbp = current_sbp  # Update last_sbp when sbp changes
            elif (current_price < entry_price) and (current_sbp == 0):  # sbp change, not profitable
                og_df.at[i, 'Trade'] = 'Exit Long'
                og_df.at[i, 'Reason'] = 'SBP change'
                active_trade = None
                entry_price = None
                highest_price = None
                last_sbp = current_sbp  # Update last_sbp when sbp changes
    
        # Handle exit conditions for short trades
        elif active_trade == 'short':
            lowest_price = min(lowest_price, current_price)
    
            if (current_price > entry_price) and (entry_price / current_price - 1 <= -0.05):  # Stoploss
                og_df.at[i, 'Trade'] = 'Exit Short'
                og_df.at[i, 'Reason'] = 'Stoploss'
                active_trade = None
                entry_price = None
                lowest_price = None
            elif (current_price < entry_price) and (entry_price / current_price - 1 >= 0.05):  # Takeprofit
                og_df.at[i, 'Trade'] = 'Exit Short'
                og_df.at[i, 'Reason'] = 'Takeprofit'
                active_trade = None
                entry_price = None
                lowest_price = None
            elif (current_sbp == 0 and current_price < entry_price):  # Start trailing stoploss when sbp changes
                trailing_stoploss_price = lowest_price * 1.01
                if current_price > trailing_stoploss_price:
                    og_df.at[i, 'Trade'] = 'Exit Short'
                    og_df.at[i, 'Reason'] = 'Trailing stoploss'
                    active_trade = None
                    entry_price = None
                    lowest_price = None
                last_sbp = current_sbp  # Update last_sbp when sbp changes
            elif (current_price > entry_price) and (current_sbp == 0):  # sbp change, not profitable
                og_df.at[i, 'Trade'] = 'Exit Short'
                og_df.at[i, 'Reason'] = 'SBP change'
                active_trade = None
                entry_price = None
                lowest_price = None
                last_sbp = current_sbp  # Update last_sbp when sbp changes
                
    # Initialize additional columns
    og_df['Entry_Price'] = np.nan
    og_df['Exit_Price'] = np.nan
    og_df['P&L'] = np.nan
    og_df['P&L_Cost'] = np.nan
    
    for i in range(1, len(og_df)):
        if pd.notna(og_df['Trade'].iloc[i]) and 'Enter' in og_df['Trade'].iloc[i]:
            entry_price = og_df['Close'].iloc[i]
            active_trade = og_df['Trade'].iloc[i].split()[1].lower()  # 'long' or 'short'
            og_df.at[i, 'Entry_Price'] = entry_price
        
        elif pd.notna(og_df['Trade'].iloc[i]) and 'Exit' in og_df['Trade'].iloc[i]:
            reason = og_df['Reason'].iloc[i]
            if reason == 'Takeprofit':
                exit_price = entry_price * 1.10 if active_trade == 'long' else entry_price * 0.95
            elif reason == 'Stoploss':
                exit_price = entry_price * 0.95 if active_trade == 'long' else entry_price * 1.05
            elif reason == 'Trailing stoploss' or reason == 'SBP change':
                exit_price = og_df['Close'].iloc[i]
    
            og_df.at[i, 'Exit_Price'] = exit_price
            
            # Calculate P&L without costs
            if active_trade == 'long':
                pnl = exit_price - entry_price
            elif active_trade == 'short':
                pnl = entry_price - exit_price
            
            og_df.at[i, 'P&L'] = pnl
            
            # Calculate P&L with costs
            if active_trade == 'long':
                pnl_cost = 0.997 * exit_price - 1.003 * entry_price
            elif active_trade == 'short':
                pnl_cost = 0.997 * entry_price - 1.003 * exit_price
            
            og_df.at[i, 'P&L_Cost'] = pnl_cost
    
    
    return og_df


tickers = ["ABB", "AUBANK", "ABBOTINDIA", "ADANIPORTS", "ABFRL", "AMBUJACEM", "APOLLOTYRE", "ASIANPAINT",
           "ATUL", "AXISBANK", "BAJFINANCE", "BALKRISIND", "BANDHANBNK", "BATAINDIA", "BEL", "BHEL", "BHARTIARTL",
           "BSOFT", "BRITANNIA", "CANBK", "CHOLAFIN", "CUB", "COFORGE", "CONCOR", "CROMPTON", "DLF",
           "DALBHARAT", "DIVISLAB", "LALPATHLAB", "EICHERMOT", "EXIDEIND", "GAIL", "GLS", "GODREJPROP",
           "GRASIM", "GNFC", "HDFCAMC", "HDFCLIFE", "HEROMOTOCO", "HAL", "HINDPETRO", "ICICIBANK",
           "ICICIPRULI", "IDFC", "INDIACEM", "IEX", "IOC", "IGL", "INDUSINDBK", "INFY", "IPCALAB", "JSWSTEEL",
           "JUBLFOOD", "LTF", "LICHSGFIN", "LT", "LUPIN", "MGL", "M&M", "MARICO", "MFSL", "MPHASIS", "MUTHOOTFIN",
           "NTPC", "NAVINFLUOR", "ONGC", "PIIND", "PAGEIND", "PETRONET", "PEL", "PFC", "PNB", "RECLTD", "SBICARD",
           "SRF", "SHREECEM", "SIEMENS", "SAIL", "SUNTV", "TVSMOTOR", "TATACOMM", "TATACONSUM", "TATAPOWER", "TECHM",
           "TITAN", "TRENT", "ULTRACEMCO",  "IDEA", "WIPRO", "ZYDUSLIFE", "ACC", "AARTIIND", "ADANIENT", "ABCAPITAL",
           "ALKEM", "APOLLOHOSP", "ASHOKLEY", "ASTRAL", "AUROPHARMA", "BAJAJ-AUTO", "BAJAJFINSV", "BALRAMCHIN",
           "BANKBARODA", "BERGEPAINT", "BHARATFORG", "BPCL", "BIOCON", "BOSCHLTD", "CANFINHOME", "CHAMBLFERT",
           "CIPLA", "COALINDIA", "COLPAL", "COROMANDEL", "CUMMINSIND", "DABUR", "DEEPAKNTR", "DIXON", "DRREDDY",
          "ESCORTS", "FEDERALBNK", "GMRINFRA", "GODREJCP", "GRANULES", "GUJGASLTD", "HCLTECH", "HDFCBANK", 
          "HAVELLS", "HINDALCO", "HINDCOPPER", "HINDUNILVR", "ICICIGI", "IDFCFIRSTB", "ITC", "INDIAMART",
          "INDHOTEL", "IRCTC", "INDUSTOWER", "NAUKRI", "INDIGO", "JKCEMENT", "JINDALSTEL", "KOTAKBANK", 
          "LTTS", "LTIM", "LAURUSLABS", "MRF", "M&MFIN", "MANAPPURAM", "MARUTI", "METROPOLIS", "MCX", 
          "NMDC","NATIONALUM", "NESTLEIND", "OBEROIRLTY", "OFSS", "PVRINOX", "PERSISTENT", "PIDILITIND",
          "POLYCAB", "POWERGRID", "RBLBANK", "RELIANCE", "SBILIFE", "MOTHERSON", "SHRIRAMFIN", "SBIN",
          "SUNPHARMA", "SYNGENE", "TATACHEM", "TCS", "TATAMOTORS", "TATASTEEL", "RAMCOCEM", "TORNTPHARM",
          "UPL", "UBL", "VEDL", "VOLTAS", "ZEEL"] #"UNITDSPR" - 2023


start = dt.date(2018, 1, 1)
end = dt.date(2023, 12, 31)

tickers = [ticker + ".NS" for ticker in tickers]

ohlcv_data = {}
for ticker in tickers:
    ohlcv_data[ticker] = yf.download(ticker, start= start, end = end)

index = yf.download(tickers="^NSEI", start = start, end = end)


# =============================================================================
# # Function to invert a dataframe
# def invert_dataframe(df):
#     df = df[::-1].reset_index(drop=True)
#     df.index = pd.date_range(start=df.index.min(), periods=len(df), freq='B')
#     return df
# 
# # Invert each dataframe in ohlcv_data
# inverted_ohlcv_data = {ticker: invert_dataframe(df) for ticker, df in ohlcv_data.items()}
# 
# # Invert the index dataframe
# inverted_index = invert_dataframe(index)
# ohlcv_data = inverted_ohlcv_data
# index = inverted_index
# =============================================================================
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress the specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")

# Suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None

strategy_dfs = {} 

start_time = time.time()
no_stocks = len(tickers)

test_dfs = {}
for j in range(len(tickers)):
    if j % 10 == 0 or j == no_stocks - 1:
       elapsed_time = time.time() - start_time
       avg_time_per_iteration = elapsed_time / (j + 1)
       estimated_time_remaining = avg_time_per_iteration * (no_stocks - j - 1)
       
       print(f"Processing realization {j+1}/{no_stocks} | Elapsed: {elapsed_time:.2f} seconds | ETR: {estimated_time_remaining:.2f} seconds\n", end='\r')
    
    data = ohlcv_data[tickers[j]]["Adj Close"]
    S = data.to_numpy()
   
    # Splitting the data into test and training segments
    train_data = S[:252]
    test_data = S[252:]
       
       
    # Initialize a DataFrame to store the test data and predicted states
    test_df = pd.DataFrame(test_data, columns = ['Close'])
    test_df['State'] = np.nan
    
    # Initialize rolling window
    window_size = 252
    
    # Initial training of the HMM on the first window
    initial_window = S[:window_size]
    initial_window = initial_window.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
    model.fit(initial_window)
 
    # Iterate through the test data using a rolling window
    for i in range(252,len(S)):
        # Define the current window
        current_window = S[i-window_size:i]
        current_window_reshaped = current_window.reshape(-1, 1)
        with suppress_output():
        # Train the HMM on the current window
            model.fit(current_window_reshaped)
        
        # Predict the state for the entire window
        predicted_states = model.predict(current_window_reshaped)
        
        psi_original = hf.calculate_psi(current_window, predicted_states)
        
        # Invert the states
        inverted_states = 1 - predicted_states
        
        # Calculate psi for inverted states
        psi_inverted = hf.calculate_psi(current_window, inverted_states)
        
        # Choose the regime with the lowest psi
        if psi_inverted < psi_original:
            consistent_states = inverted_states
        else:
            consistent_states = predicted_states
        # Predict the state for the last data point in the current window
        hidden_state = consistent_states[-1]
        # Store the predicted state in the DataFrame
        test_df.iloc[i - window_size, 1] = hidden_state
        
 
    test_df['Smoothed_State'] = smooth_states_with_ema(test_df['State'], span=9, threshold=0.5)

    MACD, signal = hf.calculate_macd( pd.Series(S), slow = 52, fast = 26, signal = 18)
    test_df["MACD"] = MACD[252:].values
    test_df["Signal"] = signal[252:].values
    
    results = og(test_df.copy())
    
    strategy_dfs[tickers[j]] = results.set_index(data.iloc[252:].index)
# =============================================================================
# 
# for j in range(len(tickers)):
#     data = ohlcv_data[tickers[j]]["Close"]
#     test_df = test_dfs[tickers[j]]
#     results = og(test_df.copy())
#     strategy_dfs[tickers[j]] = results.set_index(data.iloc[252:].index)
# 
# =============================================================================
# Get a list of all unique dates from all dataframes
all_dates = sorted(set(date for df in strategy_dfs.values() for date in df.index))

# Create an empty positions matrix with all dates as the index and tickers as columns
positions_matrix = pd.DataFrame(0, index=all_dates, columns=strategy_dfs.keys())

# Create an empty dictionary to hold the P&L_cost lists for each ticker
pnl_cost_dict = {ticker: [] for ticker in strategy_dfs.keys()}

# Iterate through each ticker and its dataframe
for ticker, df in strategy_dfs.items():
    current_position = 0  # 0 indicates no position, 1 for long, -1 for short
    
    for date, row in df.iterrows():
        trade = row['Trade']
        
        if trade == 'Enter Long':
            current_position = 1
        elif trade == 'Exit Long':
            current_position = 0
            pnl_cost_dict[ticker].append(row['P&L_Cost'])
        elif trade == 'Enter Short':
            current_position = -1
        elif trade == 'Exit Short':
            current_position = 0
            pnl_cost_dict[ticker].append(row['P&L_Cost'])
        
        # Update the positions matrix
        positions_matrix.at[date, ticker] = current_position

# Forward fill the positions matrix to propagate the positions to all dates
positions_matrix = positions_matrix.ffill().fillna(0).astype(int)


# Initialize ledger dataframe identical to positions matrix but with additional columns for cash and capital
ledger = positions_matrix.copy().astype(float)
ledger['Cash'] = 0.0
ledger['Capital'] = 0.0


# Set initial capital
initial_capital = 100000.0
ledger['Cash'].iloc[0] = initial_capital
ledger['Capital'].iloc[0] = initial_capital

# Track number of trades for each ticker
trade_counter = {ticker: 0 for ticker in positions_matrix.columns}
entry_allocation = {ticker: 0 for ticker in positions_matrix.columns}

# Iterate through the days in the positions matrix
for i in range(1, len(positions_matrix)):
    daily_cash = ledger['Cash'].iloc[i - 1]
    daily_capital = ledger['Capital'].iloc[i - 1]
    
    for ticker in positions_matrix.columns:
        position = positions_matrix[ticker].iloc[i]
        previous_position = positions_matrix[ticker].iloc[i - 1]
        
        # Check for short position entry
        if (position == -1 and previous_position == 0):
            current_price = strategy_dfs[ticker].at[positions_matrix.index[i], 'Close']
            allocation = 0.05 * daily_capital  # Allocate 5% of capital
            if daily_cash < allocation:
                allocation = daily_cash if daily_cash > 0 else 0
            
            num_shares = -(allocation // current_price)
            if abs(num_shares) > 0:
                daily_cash -= abs(num_shares) * current_price
            
            ledger.at[positions_matrix.index[i], ticker] = num_shares 
            entry_allocation[ticker] = abs(num_shares*current_price)
            
        # Check for long position entry
        elif (position == 1 and previous_position == 0):
            current_price = strategy_dfs[ticker].at[positions_matrix.index[i], 'Close']
            allocation = 0.05 * daily_capital  # Allocate 5% of capital
            if daily_cash < allocation:
                allocation = daily_cash if daily_cash > 0 else 0
            
            num_shares = allocation // current_price
            if abs(num_shares) > 0:
                daily_cash -= num_shares * current_price
            
            ledger.at[positions_matrix.index[i], ticker] = num_shares 
            entry_allocation[ticker] = abs(num_shares*current_price)
        
        # Update cash and capital for exiting trades
        elif position == 0 and previous_position != 0:
            num_shares = ledger.at[positions_matrix.index[i-1], ticker]
            try:
                trade_pnl = pnl_cost_dict[ticker][trade_counter[ticker]]
                trade_counter[ticker] += 1
                pnl_value = trade_pnl * abs(num_shares)
                daily_cash += entry_allocation[ticker] + pnl_value
                daily_capital += pnl_value
                entry_allocation[ticker] = 0
            except IndexError:
                pass
        else:
            ledger.at[positions_matrix.index[i],ticker] = ledger.at[positions_matrix.index[i-1], ticker]
            
    # Update the cash and capital in the ledger
    ledger.at[positions_matrix.index[i], 'Cash'] = daily_cash
    ledger.at[positions_matrix.index[i], 'Capital'] = daily_capital

# Assuming ledger and index dataframes are available

# Ensure the index data is aligned with the strategy's trading period

index = index.iloc[252:]

# Calculate daily returns for the strategy
ledger['Daily Return'] = ledger['Capital'].pct_change()
ledger['Daily Return'].fillna(0, inplace=True)

# Calculate cumulative returns
ledger['Cumulative Return'] = (1 + ledger['Daily Return']).cumprod()

# Calculate market returns
index['Market Return'] = index['Close'].pct_change()
index['Market Return'].fillna(0, inplace=True)

# Calculate total and annualized returns
initial_capital = ledger['Capital'].iloc[0]
final_capital = ledger['Capital'].iloc[-1]
total_return = (final_capital - initial_capital) / initial_capital
annualized_return = (1 + total_return) ** (252 / len(ledger)) - 1

# Calculate annualized volatility
annualized_volatility = ledger['Daily Return'].std() * np.sqrt(252)

# Calculate Sharpe Ratio (assuming risk-free rate is 0)
sharpe_ratio = (annualized_return-0.06995)/ annualized_volatility

# Calculate max drawdown
ledger['Peak Capital'] = ledger['Capital'].cummax()
ledger['Drawdown'] = (ledger['Capital'] - ledger['Peak Capital']) / ledger['Peak Capital']
max_drawdown = ledger['Drawdown'].min()

# Calculate beta and alpha
merged_returns = pd.merge(ledger['Daily Return'], index['Market Return'], left_index=True, right_index=True)
X = sm.add_constant(merged_returns['Market Return'])
model = sm.OLS(merged_returns['Daily Return'], X).fit()
beta = model.params[1]

R_M = index['Adj Close'].iloc[-1]/index['Adj Close'].iloc[0] -1
alpha = annualized_return - 0.06995 - beta*(R_M - 0.06995)
# Generate plots separately

# Cumulative Returns Plot
plt.figure(figsize=(10, 6))
plt.plot(ledger.index, ledger['Cumulative Return'], label='Strategy')
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Daily Returns Histogram
plt.figure(figsize=(10, 6))
plt.hist(ledger['Daily Return'], bins=50, alpha=0.75)
plt.title('Daily Returns Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

# Capital Over Time Plot
plt.figure(figsize=(10, 6))
plt.plot(ledger.index, ledger['Capital'], label='Capital')
plt.title('Capital Over Time')
plt.xlabel('Date')
plt.ylabel('Capital')
plt.legend()
plt.show()

# Drawdown Plot
plt.figure(figsize=(10, 6))
plt.plot(ledger.index, ledger['Drawdown'], label='Drawdown')
plt.title('Drawdown Over Time')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.legend()
plt.show()

# Print out the summary of metrics and initial/final portfolio value
initial_value = initial_capital
final_value = final_capital

summary = f"""
Initial Portfolio Value: ${initial_value:.2f}
Final Portfolio Value: ${final_value:.2f}

Total Return: {total_return:.2%}
Annualized Return: {annualized_return:.2%}
Annualized Volatility: {annualized_volatility:.2%}
Sharpe Ratio: {sharpe_ratio:.2f}
Max Drawdown: {max_drawdown:.2%}
Alpha: {alpha:.2%}
Beta: {beta:.2f}
"""
print(summary)