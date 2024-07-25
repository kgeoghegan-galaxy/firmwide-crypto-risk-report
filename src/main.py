import pandas as pd
from datetime import datetime
import re
import os

def sum_deltas(pivot_df, trader, column_pattern):
    cols = pivot_df.filter(regex=column_pattern).columns
    return pivot_df[cols].sum().sum() if cols.shape[0] > 0 else 0

def testing_csv_processing(filepath):
    # Read the first two rows to create the header
    header_df = pd.read_csv(filepath, nrows=2, header=None)
    header_df.fillna('', inplace=True)
    header = header_df.apply(lambda x: ' '.join(map(str, x)).strip(), axis=0).tolist()

    # Ensure unique column names
    unique_header = []
    counts = {}
    for col in header:
        if col in counts:
            counts[col] += 1
            unique_header.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            unique_header.append(col)

    # Read the rest of the CSV data using the new header
    df = pd.read_csv(filepath, skiprows=2, header=None, names=unique_header)
    
    # Combine the first 5 columns to create unique column names, excluding 'nan' values
    df['combined'] = df[['Pod(L2)', 'Book(L3)', 'Underlier', 'Desk Strategy', 'Ticker']].astype(str).apply(lambda x: ' '.join(filter(lambda y: y != 'nan' and y != '', x)), axis=1)
    
    # Pivot the table to get the 'combined' columns with 'Delta' values
    pivot_df = df.pivot_table(index=None, columns='combined', values='Delta', aggfunc='first').reset_index(drop=True)
    
    # Fill down the values to handle hierarchical structure
    pivot_df.ffill(inplace=True)
    
    # Initialize a dictionary to store the final results
    result_dict = {}

    # List of traders to process, excluding Novo
    traders = ['Beimnet', 'Bouchra', 'Eduardo', 'Felman']

    for trader in traders:
        # Columns to sum for BTC, ETH, and SOL (Core, Options, and RelativeValue)
        btc_core_col = f'{trader} Crypto BTC Core BTC'
        btc_options_cols = pivot_df.filter(regex=f'{trader} Crypto BTC Options $').columns
        btc_relative_col = f'{trader} Crypto BTC RelativeValue BTC'

        eth_core_col = f'{trader} Crypto ETH Core ETH'
        eth_options_cols = pivot_df.filter(regex=f'{trader} Crypto ETH Options $').columns
        eth_relative_col = f'{trader} Crypto ETH RelativeValue ETH'

        sol_core_col = f'{trader} Crypto SOL Core SOL'
        sol_options_cols = pivot_df.filter(regex=f'{trader} Crypto SOL Options $').columns
        sol_relative_col = f'{trader} Crypto SOL RelativeValue SOL'

        # Calculate the sum for BTC, ETH, and SOL deltas
        btc_core_sum = pivot_df[btc_core_col].sum() if btc_core_col in pivot_df else 0
        btc_options_sum = pivot_df[btc_options_cols].sum().sum() if btc_options_cols.shape[0] > 0 else 0
        btc_relative_sum = pivot_df[btc_relative_col].sum() if btc_relative_col in pivot_df else 0
        btc_total_sum = btc_core_sum + btc_options_sum + btc_relative_sum

        eth_core_sum = pivot_df[eth_core_col].sum() if eth_core_col in pivot_df else 0
        eth_options_sum = pivot_df[eth_options_cols].sum().sum() if eth_options_cols.shape[0] > 0 else 0
        eth_relative_sum = pivot_df[eth_relative_col].sum() if eth_relative_col in pivot_df else 0
        eth_total_sum = eth_core_sum + eth_options_sum + eth_relative_sum

        sol_core_sum = pivot_df[sol_core_col].sum() if sol_core_col in pivot_df else 0
        sol_options_sum = pivot_df[sol_options_cols].sum().sum() if sol_options_cols.shape[0] > 0 else 0
        sol_relative_sum = pivot_df[sol_relative_col].sum() if sol_relative_col in pivot_df else 0
        sol_total_sum = sol_core_sum + sol_options_sum + sol_relative_sum

        # Initialize the Alts Delta bucket sum
        alts_core_sum = 0
        alts_options_sum = 0
        alts_relative_sum = 0

        # Iterate over all columns to find relevant Alts Delta columns
        for col in pivot_df.columns:
            if col.startswith(f'{trader} Crypto') and not col.startswith('Novo Crypto'):
                parts = col.split()
                if len(parts) == 5:
                    underlier = parts[2]
                    ticker = parts[4]
                    if underlier == ticker and underlier not in ['BTC', 'ETH', 'SOL', 'BGCI', 'GDAM1']:
                        if 'Core' in col:
                            alts_core_sum += pivot_df[col].sum()
                        elif 'Options' in col:
                            alts_options_sum += pivot_df[col].sum()
                        elif 'RelativeValue' in col:
                            alts_relative_sum += pivot_df[col].sum()

        alts_total_sum = alts_core_sum + alts_options_sum + alts_relative_sum

        # Calculate total delta for the trader
        trader_total_sum = btc_total_sum + eth_total_sum + sol_total_sum + alts_total_sum

        # Add the trader's deltas to the result dictionary if they are non-zero
        if btc_core_sum != 0:
            result_dict[f'{trader} BTC Core Delta'] = btc_core_sum
        if btc_options_sum != 0:
            result_dict[f'{trader} BTC Options Delta'] = btc_options_sum
        if btc_relative_sum != 0:
            result_dict[f'{trader} BTC RelativeValue Delta'] = btc_relative_sum
        if btc_total_sum != 0:
            result_dict[f'{trader} BTC Delta'] = btc_total_sum

        if eth_core_sum != 0:
            result_dict[f'{trader} ETH Core Delta'] = eth_core_sum
        if eth_options_sum != 0:
            result_dict[f'{trader} ETH Options Delta'] = eth_options_sum
        if eth_relative_sum != 0:
            result_dict[f'{trader} ETH RelativeValue Delta'] = eth_relative_sum
        if eth_total_sum != 0:
            result_dict[f'{trader} ETH Delta'] = eth_total_sum

        if sol_core_sum != 0:
            result_dict[f'{trader} SOL Core Delta'] = sol_core_sum
        if sol_options_sum != 0:
            result_dict[f'{trader} SOL Options Delta'] = sol_options_sum
        if sol_relative_sum != 0:
            result_dict[f'{trader} SOL RelativeValue Delta'] = sol_relative_sum
        if sol_total_sum != 0:
            result_dict[f'{trader} SOL Delta'] = sol_total_sum

        if alts_core_sum != 0:
            result_dict[f'{trader} Alts Core Delta'] = alts_core_sum
        if alts_options_sum != 0:
            result_dict[f'{trader} Alts Options Delta'] = alts_options_sum
        if alts_relative_sum != 0:
            result_dict[f'{trader} Alts RelativeValue Delta'] = alts_relative_sum
        if alts_total_sum != 0:
            result_dict[f'{trader} Alts Delta'] = alts_total_sum

        if trader_total_sum != 0:
            result_dict[f'{trader} Delta'] = trader_total_sum

    # Process Novo separately
    novo_dict = {}

    # Columns to sum for BTC, ETH, and SOL (Core, ETF, Options)
    novo_btc_core_col = 'Novo Crypto BTC Core'
    novo_btc_etf_cols = pivot_df.filter(regex='Novo Crypto BTC ETF $').columns
    novo_btc_options_cols = pivot_df.filter(regex='Novo Crypto BTC Options $').columns

    novo_eth_core_col = 'Novo Crypto ETH Core'
    novo_eth_etf_cols = pivot_df.filter(regex='Novo Crypto ETH ETF $').columns
    novo_eth_options_cols = pivot_df.filter(regex='Novo Crypto ETH Options $').columns

    novo_sol_core_col = 'Novo Crypto SOL Core'
    novo_sol_etf_cols = pivot_df.filter(regex='Novo Crypto SOL ETF $').columns
    novo_sol_locked_col = 'Novo Crypto Locked'
    novo_sol_options_cols = pivot_df.filter(regex='Novo Crypto SOL Options $').columns

    # Calculate the sum for BTC, ETH, and SOL deltas
    novo_btc_core_sum = pivot_df[novo_btc_core_col].sum() if novo_btc_core_col in pivot_df else 0
    novo_btc_etf_sum = pivot_df[novo_btc_etf_cols].sum().sum() if len(novo_btc_etf_cols) > 0 else 0
    novo_btc_options_sum = pivot_df[novo_btc_options_cols].sum().sum() if len(novo_btc_options_cols) > 0 else 0
    novo_btc_total_sum = novo_btc_core_sum + novo_btc_etf_sum + novo_btc_options_sum

    novo_eth_core_sum = pivot_df[novo_eth_core_col].sum() if novo_eth_core_col in pivot_df else 0
    novo_eth_etf_sum = pivot_df[novo_eth_etf_cols].sum().sum() if len(novo_eth_etf_cols) > 0 else 0
    novo_eth_options_sum = pivot_df[novo_eth_options_cols].sum().sum() if len(novo_eth_options_cols) > 0 else 0
    novo_eth_total_sum = novo_eth_core_sum + novo_eth_etf_sum + novo_eth_options_sum

    novo_sol_core_sum = pivot_df[novo_sol_core_col].sum() if novo_sol_core_col in pivot_df else 0
    novo_sol_etf_sum = pivot_df[novo_sol_etf_cols].sum().sum() if len(novo_sol_etf_cols) > 0 else 0
    novo_sol_locked_sum = pivot_df[novo_sol_locked_col].sum() if novo_sol_locked_col in pivot_df else 0
    novo_sol_options_sum = pivot_df[novo_sol_options_cols].sum().sum() if len(novo_sol_options_cols) > 0 else 0
    novo_sol_total_sum = novo_sol_core_sum + novo_sol_etf_sum + novo_sol_locked_sum + novo_sol_options_sum

    # Initialize the Alts Delta bucket sum for Novo
    novo_alts_sum = 0
    for col in pivot_df.columns:
        if col.startswith('Novo Crypto') and 'Core' in col and 'Options' in col and 'RelativeValue' in col:
            parts = col.split()
            underlier = parts[2]
            ticker = parts[4]
            if underlier == ticker and underlier not in ['BTC', 'ETH', 'SOL', 'BGCI', 'GDAM1']:
                novo_alts_sum += pivot_df[col].sum()

    novo_total_sum = novo_btc_total_sum + novo_eth_total_sum + novo_sol_total_sum + novo_alts_sum

    # Add Novo's deltas to the result dictionary
    novo_dict['Novo BTC Core Delta'] = novo_btc_core_sum
    novo_dict['Novo BTC ETF Delta'] = novo_btc_etf_sum
    novo_dict['Novo BTC Options Delta'] = novo_btc_options_sum
    novo_dict['Novo BTC Delta'] = novo_btc_total_sum

    novo_dict['Novo ETH Core Delta'] = novo_eth_core_sum
    novo_dict['Novo ETH ETF Delta'] = novo_eth_etf_sum
    novo_dict['Novo ETH Options Delta'] = novo_eth_options_sum
    novo_dict['Novo ETH Delta'] = novo_eth_total_sum

    novo_dict['Novo SOL Core Delta'] = novo_sol_core_sum
    novo_dict['Novo SOL ETF Delta'] = novo_sol_etf_sum
    novo_dict['Novo SOL Locked Delta'] = novo_sol_locked_sum
    novo_dict['Novo SOL Options Delta'] = novo_sol_options_sum
    novo_dict['Novo SOL Delta'] = novo_sol_total_sum

    novo_dict['Novo Alts Delta'] = novo_alts_sum
    novo_dict['Novo Delta'] = novo_total_sum

    # Add Novo's deltas to the result dictionary
    result_dict.update(novo_dict)

    # Sum all traders' deltas by asset grouping and total
    summary_dict = {
        'GDLP ex-Novo BTC Delta': 0,
        'GDLP ex-Novo ETH Delta': 0,
        'GDLP ex-Novo SOL Delta': 0,
        'GDLP ex-Novo Alts Delta': 0,
        'GDLP ex-Novo Delta': 0,
        'GDLP ex-Novo ex-Alts Delta': 0
    }

    for key, value in result_dict.items():
        if 'BTC Delta' in key and 'Novo' not in key:
            summary_dict['GDLP ex-Novo BTC Delta'] += value
        elif 'ETH Delta' in key and 'Novo' not in key:
            summary_dict['GDLP ex-Novo ETH Delta'] += value
        elif 'SOL Delta' in key and 'Novo' not in key:
            summary_dict['GDLP ex-Novo SOL Delta'] += value
        elif 'Alts Delta' in key and 'Novo' not in key:
            summary_dict['GDLP ex-Novo Alts Delta'] += value
        elif 'Delta' in key and 'Novo' not in key:
            summary_dict['GDLP ex-Novo Delta'] += value
            if 'Alts' not in key:
                summary_dict['GDLP ex-Novo ex-Alts Delta'] += value

    # Combine GDLP ex-Novo and Novo to get GDLP Delta
    gdlp_delta_dict = {
        'GDLP BTC Total Delta': summary_dict['GDLP ex-Novo BTC Delta'] + novo_dict.get('Novo BTC Delta', 0),
        'GDLP ETH Total Delta': summary_dict['GDLP ex-Novo ETH Delta'] + novo_dict.get('Novo ETH Delta', 0),
        'GDLP SOL Total Delta': summary_dict['GDLP ex-Novo SOL Delta'] + novo_dict.get('Novo SOL Delta', 0),
        'GDLP Alts Total Delta': summary_dict.get('GDLP ex-Novo Alts Delta', 0) + novo_dict.get('Novo Alts Delta', 0),
        'GDLP Total Delta': summary_dict['GDLP ex-Novo Delta'] + novo_dict.get('Novo Delta', 0)
    }

    # Add summary_dict values to result_dict
    result_dict.update(summary_dict)
    result_dict.update(gdlp_delta_dict)

    # Create a single-row DataFrame from the combined dictionary
    final_df = pd.DataFrame(result_dict, index=['Values'])

    print(final_df)

    final_df.to_csv('output/23072024.csv', index=False)
    final_df.to_excel('output/23072024.xlsx', index=False)

    return final_df

csv_filepath = 'data/GDLP GRE Data 23072024.csv'

# Process the CSV file and print the results
testing_csv_processing(csv_filepath)
