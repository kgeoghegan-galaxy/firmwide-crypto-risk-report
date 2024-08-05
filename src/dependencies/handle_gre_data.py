import pandas as pd
import datetime
import re
import os

class GREDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.pivot_df = None
        self.result_dict = {}
        self.traders = ['Beimnet', 'Bouchra', 'Eduardo', 'Felman']

    def get_date_from_filepath(self, filepath):
        date_match = re.search(r'(\d{8})', filepath)
    
        if date_match:
            date_str = date_match.group(1)
            print(f"Extracted date string: {date_str}")
            # Parse the date string and format it
            date_obj = datetime.datetime.strptime(date_str, '%d%m%Y')
            formatted_date = date_obj.strftime('%Y-%m-%d')
            return formatted_date
        else:
            raise ValueError("No date found in the filename.")
    
    def read_csv_with_unique_header(self):
        header_df = pd.read_csv(self.filepath, nrows=2, header=None)
        header_df.fillna('', inplace=True)
        header = header_df.apply(lambda x: ' '.join(map(str, x)).strip(), axis=0).tolist()

        unique_header = []
        counts = {}
        for col in header:
            if col in counts:
                counts[col] += 1
                unique_header.append(f"{col}_{counts[col]}")
            else:
                counts[col] = 0
                unique_header.append(col)

        self.df = pd.read_csv(self.filepath, skiprows=2, header=None, names=unique_header)
    
    def combine_columns(self):
        self.df['combined'] = self.df[['Pod(L2)', 'Book(L3)', 'Underlier', 'Desk Strategy', 'Ticker']].astype(str).apply(
            lambda x: ' '.join(filter(lambda y: y != 'nan' and y != '', x)), axis=1)
    
    def pivot_dataframe(self):
        self.pivot_df = self.df.pivot_table(index=None, columns='combined', values=['Delta', '$Gamma', '$Vega'], aggfunc='first').reset_index(drop=True)
        self.pivot_df.ffill(inplace=True)
        self.pivot_df.to_csv('../output/testing.csv')

    def sum_deltas(self, column_pattern):
        cols = self.pivot_df.filter(regex=column_pattern).columns
        return self.pivot_df[cols].sum().sum() if cols.shape[0] > 0 else 0

    def calculate_trader_deltas(self):
        for trader in self.traders:
            
            btc_core_col = f'{trader} Crypto BTC Core BTC'
            btc_options_cols = self.pivot_df.filter(regex=f'{trader} Crypto BTC Options $').columns
            btc_relative_col = f'{trader} Crypto BTC RelativeValue BTC'

            eth_core_col = f'{trader} Crypto ETH Core ETH'
            eth_options_cols = self.pivot_df.filter(regex=f'{trader} Crypto ETH Options $').columns
            eth_relative_col = f'{trader} Crypto ETH RelativeValue ETH'

            sol_core_col = f'{trader} Crypto SOL Core SOL'
            sol_options_cols = self.pivot_df.filter(regex=f'{trader} Crypto SOL Options $').columns
            sol_relative_col = f'{trader} Crypto SOL RelativeValue SOL'

            btc_core_sum = self.pivot_df[btc_core_col].sum() if btc_core_col in self.pivot_df else 0
            btc_options_sum = self.pivot_df.loc[2,btc_options_cols].sum() if btc_options_cols.shape[0] > 0 else 0
            btc_relative_sum = self.pivot_df[btc_relative_col].sum() if btc_relative_col in self.pivot_df else 0
            btc_total_sum = btc_core_sum + btc_options_sum + btc_relative_sum

            eth_core_sum = self.pivot_df[eth_core_col].sum() if eth_core_col in self.pivot_df else 0
            eth_options_sum = self.pivot_df.loc[2, eth_options_cols].sum()if eth_options_cols.shape[0] > 0 else 0
            eth_relative_sum = self.pivot_df[eth_relative_col].sum() if eth_relative_col in self.pivot_df else 0
            eth_total_sum = eth_core_sum + eth_options_sum + eth_relative_sum

            sol_core_sum = self.pivot_df[sol_core_col].sum() if sol_core_col in self.pivot_df else 0
            sol_options_sum = self.pivot_df.loc[2, sol_options_cols].sum() if sol_options_cols.shape[0] > 0 else 0
            sol_relative_sum = self.pivot_df[sol_relative_col].sum() if sol_relative_col in self.pivot_df else 0
            sol_total_sum = sol_core_sum + sol_options_sum + sol_relative_sum

            alts_core_sum = 0
            alts_options_sum = 0
            alts_relative_sum = 0

            for col in self.pivot_df.columns:
                if col.startswith(f'{trader} Crypto') and not col.startswith('Novo Crypto'):
                    parts = col.split()
                    if len(parts) == 5:
                        underlier = parts[2]
                        ticker = parts[4]
                        if underlier == ticker and underlier not in ['BTC', 'ETH', 'SOL', 'BGCI', 'GDAM1']:
                            if 'Core' in col:
                                alts_core_sum += self.pivot_df[col][2]
                                print(self.pivot_df[col][2])
                            elif 'Options' in col:
                                alts_options_sum += self.pivot_df[col][2]
                            elif 'RelativeValue' in col:
                                alts_relative_sum += self.pivot_df[col][2]

            alts_total_sum = alts_core_sum + alts_options_sum + alts_relative_sum
            trader_total_sum = btc_total_sum + eth_total_sum + sol_total_sum + alts_total_sum

            if btc_core_sum != 0:
                self.result_dict[f'{trader} BTC Core Delta'] = btc_core_sum
            if btc_options_sum != 0:
                self.result_dict[f'{trader} BTC Options Delta'] = btc_options_sum
            if btc_relative_sum != 0:
                self.result_dict[f'{trader} BTC RelativeValue Delta'] = btc_relative_sum
            if btc_total_sum != 0:
                self.result_dict[f'{trader} BTC Delta'] = btc_total_sum

            if eth_core_sum != 0:
                self.result_dict[f'{trader} ETH Core Delta'] = eth_core_sum
            if eth_options_sum != 0:
                self.result_dict[f'{trader} ETH Options Delta'] = eth_options_sum
            if eth_relative_sum != 0:
                self.result_dict[f'{trader} ETH RelativeValue Delta'] = eth_relative_sum
            if eth_total_sum != 0:
                self.result_dict[f'{trader} ETH Delta'] = eth_total_sum

            if sol_core_sum != 0:
                self.result_dict[f'{trader} SOL Core Delta'] = sol_core_sum
            if sol_options_sum != 0:
                self.result_dict[f'{trader} SOL Options Delta'] = sol_options_sum
            if sol_relative_sum != 0:
                self.result_dict[f'{trader} SOL RelativeValue Delta'] = sol_relative_sum
            if sol_total_sum != 0:
                self.result_dict[f'{trader} SOL Delta'] = sol_total_sum

            if alts_core_sum != 0:
                self.result_dict[f'{trader} Alts Core Delta'] = alts_core_sum
            if alts_options_sum != 0:
                self.result_dict[f'{trader} Alts Options Delta'] = alts_options_sum
            if alts_relative_sum != 0:
                self.result_dict[f'{trader} Alts RelativeValue Delta'] = alts_relative_sum
            if alts_total_sum != 0:
                self.result_dict[f'{trader} Alts Delta'] = alts_total_sum

            if trader_total_sum != 0:
                self.result_dict[f'{trader} Delta'] = trader_total_sum

    def calculate_novo_deltas(self):
        novo_dict = {}
        novo_btc_core_col = 'Novo Crypto BTC Core'
        novo_btc_etf_cols = self.pivot_df.filter(regex='Novo Crypto BTC ETF $').columns
        novo_btc_options_cols = self.pivot_df.filter(regex='Novo Crypto BTC Options $').columns

        novo_eth_core_col = 'Novo Crypto ETH Core'
        novo_eth_etf_cols = self.pivot_df.filter(regex='Novo Crypto ETH ETF $').columns
        novo_eth_options_cols = self.pivot_df.filter(regex='Novo Crypto ETH Options $').columns

        novo_sol_core_col = 'Novo Crypto SOL Core'
        novo_sol_etf_cols = self.pivot_df.filter(regex='Novo Crypto SOL ETF $').columns
        novo_sol_locked_col = 'Novo Crypto SOL Locked'
        novo_sol_ftx_col = 'Novo Crypto FTX_SOL Locked'
        novo_sol_options_cols = self.pivot_df.filter(regex='Novo Crypto SOL Options $').columns

        novo_btc_core_sum = self.pivot_df[novo_btc_core_col].sum() if novo_btc_core_col in self.pivot_df else 0
        novo_btc_etf_sum = self.pivot_df[novo_btc_etf_cols].sum().sum() if len(novo_btc_etf_cols) > 0 else 0
        novo_btc_options_sum = self.pivot_df.loc[2, novo_btc_options_cols].sum() if len(novo_btc_options_cols) > 0 else 0
        novo_btc_total_sum = novo_btc_core_sum + novo_btc_etf_sum + novo_btc_options_sum

        novo_eth_core_sum = self.pivot_df[novo_eth_core_col].sum() if novo_eth_core_col in self.pivot_df else 0
        novo_eth_etf_sum = self.pivot_df[novo_eth_etf_cols].sum().sum() if len(novo_eth_etf_cols) > 0 else 0
        novo_eth_options_sum = self.pivot_df.loc[2, novo_eth_options_cols].sum() if len(novo_eth_options_cols) > 0 else 0
        novo_eth_total_sum = novo_eth_core_sum + novo_eth_etf_sum + novo_eth_options_sum

        novo_sol_core_sum = self.pivot_df[novo_sol_core_col].sum() if novo_sol_core_col in self.pivot_df else 0
        novo_sol_etf_sum = self.pivot_df[novo_sol_etf_cols].sum().sum() if len(novo_sol_etf_cols) > 0 else 0
        novo_sol_locked_sum = self.pivot_df[novo_sol_locked_col].sum() if novo_sol_locked_col in self.pivot_df else 0
        novo_sol_ftx_sum = self.pivot_df[novo_sol_ftx_col].sum() if novo_sol_ftx_col in self.pivot_df else 0
        novo_sol_options_sum = self.pivot_df.loc[2, novo_sol_options_cols].sum() if len(novo_sol_options_cols) > 0 else 0

        novo_sol_total_sum = novo_sol_core_sum + novo_sol_etf_sum + novo_sol_locked_sum + novo_sol_ftx_sum + novo_sol_options_sum 

        alts_core_sum = 0
        alts_options_sum = 0
        alts_relative_sum = 0

        for col in self.pivot_df.columns:
            if col.startswith('Novo Crypto'):
                parts = col.split()
                if len(parts) == 5:
                    underlier = parts[2]
                    ticker = parts[4]
                    if underlier == ticker and underlier not in ['BTC', 'ETH', 'SOL', 'BGCI', 'GDAM1']:
                        if 'Core' in col or 'SpecialSits' in col:
                            alts_core_sum += self.pivot_df[col][2]
                        elif 'Options' in col:
                            alts_options_sum += self.pivot_df[col][2]
                        elif 'RelativeValue' in col:
                            alts_relative_sum += self.pivot_df[col][2]

        alts_total_sum = alts_core_sum + alts_options_sum + alts_relative_sum

        novo_total_sum = novo_btc_total_sum + novo_eth_total_sum + novo_sol_total_sum + alts_total_sum

        # btc_price = self.pivot_df.loc[3, novo_btc_core_col].sum() if novo_btc_core_col in self.pivot_df else 0
        # eth_price = self.pivot_df.loc[3, novo_eth_core_col].sum() if novo_eth_core_col in self.pivot_df else 0
        # sol_price = self.pivot_df.loc[3, novo_sol_core_col].sum() if novo_sol_core_col in self.pivot_df else 0

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
        novo_dict['Novo SOL Locked SOL Delta'] = novo_sol_locked_sum
        novo_dict['Novo SOL Locked FTX_SOL Delta'] = novo_sol_ftx_sum
        novo_dict['Novo SOL Options Delta'] = novo_sol_options_sum
        novo_dict['Novo SOL Delta'] = novo_sol_total_sum

        novo_dict['Novo Alts Core Delta'] = alts_core_sum
        novo_dict['Novo Alts Options Delta'] = alts_options_sum
        novo_dict['Novo Alts RelativeValue Delta'] = alts_relative_sum
        novo_dict['Novo Alts Delta'] = alts_total_sum

        novo_dict['Novo Delta'] = novo_total_sum

        # novo_dict['BTC Price'] = btc_price
        # novo_dict['ETH Price'] = eth_price
        # novo_dict['SOL Price'] = sol_price

        self.result_dict.update(novo_dict)

    def calculate_trader_greeks(self):
        for trader in self.traders:
            btc_options_cols = self.pivot_df.filter(regex=f'{trader} Crypto BTC Options $').columns
            btc_gamma_sum = self.pivot_df.loc[0, btc_options_cols].sum() if btc_options_cols.shape[0] > 0 else 0
            btc_vega_sum = self.pivot_df.loc[1, btc_options_cols].sum() if btc_options_cols.shape[0] > 0 else 0
            eth_options_cols = self.pivot_df.filter(regex=f'{trader} Crypto ETH Options $').columns
            eth_gamma_sum = self.pivot_df.loc[0, eth_options_cols].sum() if eth_options_cols.shape[0] > 0 else 0
            eth_vega_sum = self.pivot_df.loc[1, eth_options_cols].sum() if eth_options_cols.shape[0] > 0 else 0
            sol_options_cols = self.pivot_df.filter(regex=f'{trader} Crypto SOL Options $').columns
            sol_gamma_sum = self.pivot_df.loc[0, sol_options_cols].sum() if sol_options_cols.shape[0] > 0 else 0
            sol_vega_sum = self.pivot_df.loc[1, sol_options_cols].sum() if sol_options_cols.shape[0] > 0 else 0

            if btc_gamma_sum != 0:
                self.result_dict[f'{trader} BTC Options Gamma'] = btc_gamma_sum
            if btc_vega_sum != 0:
                self.result_dict[f'{trader} BTC Options Vega'] = btc_vega_sum

            if eth_gamma_sum != 0:
                self.result_dict[f'{trader} ETH Options Gamma'] = eth_gamma_sum
            if eth_vega_sum != 0:
                self.result_dict[f'{trader} ETH Options Vega'] = eth_vega_sum

            if sol_gamma_sum != 0:
                self.result_dict[f'{trader} SOL Options Gamma'] = sol_gamma_sum
            if sol_vega_sum != 0:
                self.result_dict[f'{trader} SOL Options Vega'] = sol_vega_sum

    def calculate_novo_greeks(self):
        btc_options_cols = self.pivot_df.filter(regex='Novo Crypto BTC Options $').columns
        btc_gamma_sum = self.pivot_df.loc[0, btc_options_cols].sum() if btc_options_cols.shape[0] > 0 else 0
        btc_vega_sum = self.pivot_df.loc[1, btc_options_cols].sum() if btc_options_cols.shape[0] > 0 else 0
        eth_options_cols = self.pivot_df.filter(regex='Novo Crypto ETH Options $').columns
        eth_gamma_sum = self.pivot_df.loc[0, eth_options_cols].sum() if eth_options_cols.shape[0] > 0 else 0
        eth_vega_sum = self.pivot_df.loc[1, eth_options_cols].sum() if eth_options_cols.shape[0] > 0 else 0
        sol_options_cols = self.pivot_df.filter(regex='Novo Crypto SOL Options $').columns
        sol_gamma_sum = self.pivot_df.loc[0, sol_options_cols].sum() if sol_options_cols.shape[0] > 0 else 0
        sol_vega_sum = self.pivot_df.loc[1, sol_options_cols].sum() if sol_options_cols.shape[0] > 0 else 0

        if btc_gamma_sum != 0:
            self.result_dict['Novo BTC Options Gamma'] = btc_gamma_sum
        if btc_vega_sum != 0:
            self.result_dict['Novo BTC Options Vega'] = btc_vega_sum

        if eth_gamma_sum != 0:
            self.result_dict['Novo ETH Options Gamma'] = eth_gamma_sum
        if eth_vega_sum != 0:
            self.result_dict['Novo ETH Options Vega'] = eth_vega_sum

        if sol_gamma_sum != 0:
            self.result_dict['Novo SOL Options Gamma'] = sol_gamma_sum
        if sol_vega_sum != 0:
            self.result_dict['Novo SOL Options Vega'] = sol_vega_sum

    def calculate_summary(self):
        summary_dict = {
            'GDLP ex-Novo BTC Delta': 0,
            'GDLP ex-Novo ETH Delta': 0,
            'GDLP ex-Novo SOL Delta': 0,
            'GDLP ex-Novo Alts Delta': 0,
            'GDLP ex-Novo Delta': 0,
            'GDLP ex-Novo ex-Alts Delta': 0,
            'GDLP ex-Novo BTC Gamma': 0,
            'GDLP ex-Novo ETH Gamma': 0,
            'GDLP ex-Novo SOL Gamma': 0,
            'GDLP ex-Novo Alts Gamma': 0,
            'GDLP ex-Novo Gamma': 0,
            'GDLP ex-Novo BTC Vega': 0,
            'GDLP ex-Novo ETH Vega': 0,
            'GDLP ex-Novo SOL Vega': 0,
            'GDLP ex-Novo Alts Vega': 0,
            'GDLP ex-Novo Vega': 0,
        }

        for key, value in self.result_dict.items():
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

            if 'BTC Options Gamma' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo BTC Gamma'] += value
            elif 'ETH Options Gamma' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo ETH Gamma'] += value
            elif 'SOL Options Gamma' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo SOL Gamma'] += value
            elif 'Alts Options Gamma' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo Alts Gamma'] += value
            elif 'Gamma' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo Gamma'] += value

            if 'BTC Options Vega' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo BTC Vega'] += value
            elif 'ETH Options Vega' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo ETH Vega'] += value
            elif 'SOL Options Vega' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo SOL Vega'] += value
            elif 'Alts Options Vega' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo Alts Vega'] += value
            elif 'Vega' in key and 'Novo' not in key:
                summary_dict['GDLP ex-Novo Vega'] += value

        gdlp_delta_dict = {
            'GDLP BTC Total Delta': summary_dict['GDLP ex-Novo BTC Delta'] + self.result_dict.get('Novo BTC Delta', 0),
            'GDLP ETH Total Delta': summary_dict['GDLP ex-Novo ETH Delta'] + self.result_dict.get('Novo ETH Delta', 0),
            'GDLP SOL Total Delta': summary_dict['GDLP ex-Novo SOL Delta'] + self.result_dict.get('Novo SOL Delta', 0),
            'GDLP Alts Total Delta': summary_dict.get('GDLP ex-Novo Alts Delta', 0) + self.result_dict.get('Novo Alts Delta', 0),
            'GDLP Total Delta': summary_dict['GDLP ex-Novo Delta'] + self.result_dict.get('Novo Delta', 0),
            'GDLP BTC Total Gamma': summary_dict['GDLP ex-Novo BTC Gamma'] + self.result_dict.get('Novo BTC Options Gamma', 0),
            'GDLP ETH Total Gamma': summary_dict['GDLP ex-Novo ETH Gamma'] + self.result_dict.get('Novo ETH Options Gamma', 0),
            'GDLP SOL Total Gamma': summary_dict['GDLP ex-Novo SOL Gamma'] + self.result_dict.get('Novo SOL Options Gamma', 0),
            'GDLP Alts Total Gamma': summary_dict.get('GDLP ex-Novo Alts Gamma', 0) + self.result_dict.get('Novo Alts Gamma', 0),
            'GDLP Total Gamma': summary_dict['GDLP ex-Novo Gamma'] + self.result_dict.get('Novo Gamma', 0),
            'GDLP BTC Total Vega': summary_dict['GDLP ex-Novo BTC Vega'] + self.result_dict.get('Novo BTC Options Vega', 0),
            'GDLP ETH Total Vega': summary_dict['GDLP ex-Novo ETH Vega'] + self.result_dict.get('Novo ETH Options Vega', 0),
            'GDLP SOL Total Vega': summary_dict['GDLP ex-Novo SOL Vega'] + self.result_dict.get('Novo SOL Options Vega', 0),
            'GDLP Alts Total Vega': summary_dict.get('GDLP ex-Novo Alts Vega', 0) + self.result_dict.get('Novo Alts Vega', 0),
            'GDLP Total Vega': summary_dict['GDLP ex-Novo Vega'] + self.result_dict.get('Novo Vega', 0),
        }

        summary_dict.update(gdlp_delta_dict)
        self.result_dict.update(summary_dict)


    def save_results(self, date_str):
        final_df = pd.DataFrame(self.result_dict, index=['Values'])
        print(final_df)
        final_df.to_csv(f'../output/{date_str}.csv', index=False)
        #final_df.to_excel(f'../output/{date_str}.xlsx', index=False)
        return final_df

    def process(self):
        date_str = self.get_date_from_filepath(self.filepath)
        self.read_csv_with_unique_header()
        self.combine_columns()
        self.pivot_dataframe()
        self.calculate_trader_deltas()
        self.calculate_trader_greeks()
        self.calculate_novo_deltas()
        self.calculate_novo_greeks()
        self.calculate_summary()
        return self.save_results(date_str)