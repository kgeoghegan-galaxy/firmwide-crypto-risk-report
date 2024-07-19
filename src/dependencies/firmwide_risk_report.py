from dependencies.gdlp_data_construction import FrontpageTimeSeriesConstruction
from dependencies.gdt_data_construction import parse_and_clean_gdt_data
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import datetime
import re
import os

logo_path = 'Galaxy_Lockup_Horizontal_Black.png'

# Define theme colors
COLORS = {
    'Galaxy Orange 1': '#ff6400',
    'Galaxy Orange 2': '#ffaf95',
    'Galaxy Orange 3': '#fff0dd',
    'Galaxy Yellow 1': '#ffe300',
    'Galaxy Yellow 2': '#fcf0a4',
    'Galaxy Yellow 3': '#fafadc',
    'Galaxy Green 1': '#00ce84',
    'Galaxy Green 2': '#a0ffa0',
    'Galaxy Green 3': '#e2ffd8',
    'Galaxy Blue 1': '#7dccff',
    'Galaxy Blue 2': '#b8dbff',
    'Galaxy Blue 3': '#d9ebfa',
    'Galaxy Purple 1': '#9b69ff',
    'Galaxy Purple 2': '#beaaff',
    'Galaxy Purple 3': '#ebdeff',
    'Galaxy Gray 1': '#efefef',
    'Galaxy Gray 2': '#cccccc',
    'Galaxy Gray 3': '#939598',
    'Galaxy Gray 4': '#6d6e71',
    'Galaxy Gray 5': '#323232',
    'Galaxy Black': '#000000',
    'Galaxy White': '#ffffff'
}

# Map assets to colors
asset_colors = {
    'Alts': COLORS['Galaxy Orange 1'],
    'BTC': COLORS['Galaxy Purple 1'],
    'ETH': COLORS['Galaxy Green 1'],
    'SOL': COLORS['Galaxy Blue 1'],
    'Other Digital Assets': COLORS['Galaxy Orange 1'],
    'Passive Beta': COLORS['Galaxy Orange 2'],
    'Aggregate' : COLORS['Galaxy Black']
}

# Convert colors to uppercase for Plotly
gd_colors = [color.upper() for color in COLORS.values()]

class FirmwideRiskReport:

    def __init__(self, base_directory, gdlp_filename_pattern, novo_filename_pattern, derivs_csv):
        self.base_directory = base_directory
        self.gdlp_filename_pattern = gdlp_filename_pattern
        self.novo_filename_pattern = novo_filename_pattern
        self.derivs_csv = derivs_csv

    def extract_spot_shock(self, column_name):
        match = re.search(r'Spot (-?\d+)', column_name)
        if match:
            return (float(match.group(1)) / 100)
        return 0
    
    def mstr_df_setup(self):
        ticker_groups = {                                             
            'MSTR': ['MSTR']
        }

        metrics = ['Mkt Val (ex DLOM) T-0 ($)']

        analysis = FrontpageTimeSeriesConstruction(self.base_directory)
        MSTR_position_time_series_df = analysis.collect_data(self.novo_filename_pattern, ticker_groups, metrics, preprocess_tickers=False)
        #analysis.save_to_csv('../output/MSTR_position.csv')

        MSTR_position_time_series_df = MSTR_position_time_series_df.pivot_table(index='Date', columns='Ticker', values='Value').reset_index()

        return MSTR_position_time_series_df

    def novo_df_setup(self, MSTR_position_time_series_df):
        ticker_groups = {                                             
            'BTC': ['BTC']
        }

        metrics = ['Mkt Val (ex DLOM) T-0 ($)']

        analysis = FrontpageTimeSeriesConstruction(self.base_directory)
        unedited_novo_time_series_df = analysis.collect_data(self.novo_filename_pattern, ticker_groups, metrics, preprocess_tickers=True)
        unedited_novo_time_series_df.to_csv("../output/unedited_novo_time_series.csv")

        pivoted_novo_time_series_df = unedited_novo_time_series_df.pivot_table(index='Date', columns='Ticker', values='Value').reset_index()

        # Ensure both DataFrames are aligned by Date
        pivoted_novo_time_series_df['Date'] = pd.to_datetime(pivoted_novo_time_series_df['Date'])
        MSTR_position_time_series_df['Date'] = pd.to_datetime(MSTR_position_time_series_df['Date'])

        # Filter MSTR to include only the dates that are in the Novo time series
        filtered_mstr_df = MSTR_position_time_series_df[MSTR_position_time_series_df.index.isin(pivoted_novo_time_series_df.index)]

        # Add the MSTR position to the BTC column
        pivoted_novo_time_series_df['BTC'] += filtered_mstr_df['MSTR']

        # Reset the index to include 'Date' as a column
        pivoted_novo_time_series_df.reset_index(inplace=True)

        return pivoted_novo_time_series_df

    def gdlp_ex_novo_df_setup(self, novo_time_series_df, MSTR_position_time_series_df):
        ticker_groups = {
            'BTC': ['BTC']
        }

        metrics = ['Mkt Val (ex DLOM) T-0 ($)']

        analysis = FrontpageTimeSeriesConstruction(self.base_directory)
        gdlp_time_series_df = analysis.collect_data(self.gdlp_filename_pattern, ticker_groups, metrics, preprocess_tickers=True)
        #time_series.to_csv("../output/gdlp_time_series.csv")

        pivoted_gdlp_time_series_df = gdlp_time_series_df.pivot_table(index='Date', columns='Ticker', values='Value').reset_index()
        pivoted_gdlp_time_series_df.to_csv('../output/pivoted_gdlp_df.csv')

        # Ensure 'Date' columns are in datetime format
        pivoted_gdlp_time_series_df['Date'] = pd.to_datetime(pivoted_gdlp_time_series_df['Date'])
        novo_time_series_df['Date'] = pd.to_datetime(novo_time_series_df['Date'])
        MSTR_position_time_series_df['Date'] = pd.to_datetime(MSTR_position_time_series_df['Date'])

        # Set 'Date' as the index
        pivoted_gdlp_time_series_df.set_index('Date', inplace=True)
        novo_time_series_df.set_index('Date', inplace=True)
        MSTR_position_time_series_df.set_index('Date', inplace=True)

        # Filter novo_time_series_df and MSTR_position_time_series_df to match only the dates that gdlp_time_series_df has
        filtered_novo_df = novo_time_series_df.loc[pivoted_gdlp_time_series_df.index]
        filtered_mstr_df = MSTR_position_time_series_df.loc[pivoted_gdlp_time_series_df.index]

        # Subtract the Novo positions from the relevant columns
        for ticker in ['BTC', 'SOL', 'ETH', 'Alts', 'Passive Beta']:
            if ticker in pivoted_gdlp_time_series_df.columns and ticker in filtered_novo_df.columns:
                pivoted_gdlp_time_series_df[ticker] -= filtered_novo_df[ticker]

        # # Subtract the MSTR position from the BTC column
        pivoted_gdlp_time_series_df['BTC'] += filtered_mstr_df['MSTR']

        gdlp_ex_novo_time_series_df = pivoted_gdlp_time_series_df.reset_index()

        return gdlp_ex_novo_time_series_df

    def update_agg_firmwide_greeks(self, time_series_df, agg_firmwide_greeks_df, multiplied_flag):
        # Add a flag column if it doesn't exist
        multiplied_flag = multiplied_flag

        for index, row in time_series_df.iterrows():
            date = row['Date']
            for ticker in ['BTC', 'SOL', 'ETH', 'Alts', 'Passive Beta']:
                if ticker in row and ticker != 'Date':
                    value = row[ticker]
                    aggregate_delta_column = f"Aggregate {ticker} $delta"

                    # Ensure the aggregate_delta_column exists
                    if aggregate_delta_column not in agg_firmwide_greeks_df.columns:
                        agg_firmwide_greeks_df[aggregate_delta_column] = 0

                    if date in agg_firmwide_greeks_df['Date'].values:
                        current_value = agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, aggregate_delta_column].values[0]
                        
                        # Explicitly cast values to float to ensure compatibility
                        current_value = float(current_value)
                        value = float(value)
                        
                        agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, aggregate_delta_column] = current_value + value

                    for column in agg_firmwide_greeks_df.columns:
                        if column.startswith(f"{ticker} Delta Shock - Spot"):
                            spot_shock_percent = self.extract_spot_shock(column)
                            adjusted_value = value * spot_shock_percent
                            if date in agg_firmwide_greeks_df['Date'].values:
                                if multiplied_flag is False:
                                    agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, column] *= 1000
                                agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, column] += adjusted_value

                        if column.startswith(f"{ticker} PnL Shock - Spot"):
                            spot_shock_percent = self.extract_spot_shock(column)
                            adjusted_value = value * spot_shock_percent
                            if date in agg_firmwide_greeks_df['Date'].values:
                                if multiplied_flag is False:
                                    agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, column] *= 1000
                                agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, column] += adjusted_value

        return agg_firmwide_greeks_df

    def update_agg_firmwide_greeks_with_derivs(self, derivs_df, agg_firmwide_greeks_df):
        for index, row in derivs_df.iterrows():
            date = row['Date']
            for asset in ['BTC', 'ETH', 'SOL', 'Alts']:
                delta_column = f"Derivs {asset} $delta"
                aggregate_delta_column = f"Aggregate {asset} $delta"

                # Ensure the aggregate_delta_column exists
                if aggregate_delta_column not in agg_firmwide_greeks_df.columns:
                    agg_firmwide_greeks_df[aggregate_delta_column] = 0

                if date in agg_firmwide_greeks_df['Date'].values:
                    current_value = agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, aggregate_delta_column]
                    agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, aggregate_delta_column] = current_value + row[delta_column]

        return agg_firmwide_greeks_df

    def append_deltas(self, group, asset, time_series_df, agg_firmwide_greeks_df, earliest_date):
        delta_column = f"{group} {asset} $delta"
        if delta_column not in agg_firmwide_greeks_df.columns:
            agg_firmwide_greeks_df[delta_column] = np.nan
        
        time_series_df = time_series_df[time_series_df['Date'] >= earliest_date]  # Filter by earliest date
        
        for index, row in time_series_df.iterrows():
            date = row['Date']
            if asset in row and date in agg_firmwide_greeks_df['Date'].values:
                agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, delta_column] = row[asset]
                
        return agg_firmwide_greeks_df

    def rename_derivs_columns(self, derivs_df):
        new_columns = {}
        for col in derivs_df.columns:
            if col != 'Date':
                if ' vanna' in col:
                    asset, greek = col.split(' vanna')
                    new_columns[col] = f"Derivs {asset} $vanna"
                elif ' $' in col:
                    asset, greek = col.split(' $')
                    new_columns[col] = f"Derivs {asset} ${greek}"
                else:
                    new_columns[col] = col  # Keep the original column name if it doesn't match the criteria
        derivs_df = derivs_df.rename(columns=new_columns)
        return derivs_df

    def add_aggregate_total_delta(self, agg_firmwide_greeks_df):
        agg_firmwide_greeks_df['Aggregate Total $delta'] = (
            agg_firmwide_greeks_df['Aggregate BTC $delta'].fillna(0) +
            agg_firmwide_greeks_df['Aggregate ETH $delta'].fillna(0) +
            agg_firmwide_greeks_df['Aggregate SOL $delta'].fillna(0) +
            agg_firmwide_greeks_df['Aggregate Alts $delta'].fillna(0) + 
            agg_firmwide_greeks_df['Aggregate Passive Beta $delta'].fillna(0)
        )
        return agg_firmwide_greeks_df

    def add_btc_price_and_mining_constant(self, agg_firmwide_greeks_df):
        ticker_groups = {'BTC': ['BTC']}
        metrics = ['Reference Price\n($) T-0']

        analysis = FrontpageTimeSeriesConstruction(self.base_directory)
        btc_price_time_series_df = analysis.collect_data(self.gdlp_filename_pattern, ticker_groups, metrics, preprocess_tickers=False)
        btc_price_time_series_df['Date'] = pd.to_datetime(btc_price_time_series_df['Date'])

        pivoted_btc_price_time_series_df = btc_price_time_series_df.pivot(index='Date', columns='Ticker', values='Value').reset_index()
        pivoted_btc_price_time_series_df = pivoted_btc_price_time_series_df.rename(columns={"BTC": "BTC Reference Price T-0"})

        pivoted_btc_price_time_series_df['BTC Reference Price T-0'] = pivoted_btc_price_time_series_df['BTC Reference Price T-0'].fillna(0)

        agg_firmwide_greeks_df = pd.merge(agg_firmwide_greeks_df, pivoted_btc_price_time_series_df[['Date', 'BTC Reference Price T-0']], on='Date', how='left')

        agg_firmwide_greeks_df['BTC Mining PB'] = agg_firmwide_greeks_df['BTC Reference Price T-0'] * 270
        agg_firmwide_greeks_df['Aggregate Passive Beta $delta'] = agg_firmwide_greeks_df['Aggregate Passive Beta $delta'] + agg_firmwide_greeks_df['BTC Mining PB']

        return agg_firmwide_greeks_df

    def add_var_and_vol(self, agg_firmwide_greeks_df):
        ticker_groups = {'BTC': ['BTC'], 'ETH': ['ETH'], 'SOL': ['SOL']}
        metrics_var = ['Portfolio VaR95\n($)']
        metrics_vol = ['6M Vol T-0 ($)']

        analysis = FrontpageTimeSeriesConstruction(self.base_directory)
        
        # Collecting Portfolio VaR95 data
        var_time_series_df = analysis.collect_data(self.gdlp_filename_pattern, ticker_groups, metrics_var, preprocess_tickers=False)
        var_time_series_df['Date'] = pd.to_datetime(var_time_series_df['Date'])

        # Ensure no duplicate dates in var_time_series_df before pivoting
        var_time_series_df = var_time_series_df.drop_duplicates(subset=['Date', 'Ticker'])

        pivoted_var_time_series_df = var_time_series_df.pivot(index='Date', columns='Ticker', values='Value').reset_index()
        pivoted_var_time_series_df = pivoted_var_time_series_df.rename(columns={"BTC": "BTC VaR95", "ETH": "ETH VaR95", "SOL": "SOL VaR95"})
        agg_firmwide_greeks_df = pd.merge(agg_firmwide_greeks_df, pivoted_var_time_series_df[['Date', 'BTC VaR95', 'ETH VaR95', 'SOL VaR95']], on='Date', how='left')

        # Collecting 6M Vol data
        analysis = FrontpageTimeSeriesConstruction(self.base_directory)
        vol_time_series_df = analysis.collect_data(self.gdlp_filename_pattern, ticker_groups, metrics_vol, preprocess_tickers=False)
        vol_time_series_df['Date'] = pd.to_datetime(vol_time_series_df['Date'])

        # Ensure no duplicate dates in vol_time_series_df before pivoting
        vol_time_series_df = vol_time_series_df.drop_duplicates(subset=['Date', 'Ticker'])

        pivoted_vol_time_series_df = vol_time_series_df.pivot(index='Date', columns='Ticker', values='Value').reset_index()
        pivoted_vol_time_series_df = pivoted_vol_time_series_df.rename(columns={"BTC": "BTC 6m Vol", "ETH": "ETH 6m Vol", "SOL": "SOL 6m Vol"})
        agg_firmwide_greeks_df = pd.merge(agg_firmwide_greeks_df, pivoted_vol_time_series_df[['Date', 'BTC 6m Vol', 'ETH 6m Vol', 'SOL 6m Vol']], on='Date', how='left')

        return agg_firmwide_greeks_df

    def calculate_week_over_week_delta(self, agg_firmwide_greeks_df, groups, assets):
        for group in groups:
            for asset in assets:
                delta_column = f"{group} {asset} $delta"
                wow_delta_column = f"{delta_column} WoW"

                # Check if the delta column exists before proceeding
                if delta_column in agg_firmwide_greeks_df.columns:
                    agg_firmwide_greeks_df[wow_delta_column] = np.nan

                    for i in range(len(agg_firmwide_greeks_df)):
                        current_date = agg_firmwide_greeks_df.loc[i, 'Date']
                        one_week_ago = current_date - pd.Timedelta(days=7)

                        past_dates = agg_firmwide_greeks_df[agg_firmwide_greeks_df['Date'] <= one_week_ago]['Date']
                        if not past_dates.empty:
                            last_date = past_dates.max()
                            current_value = agg_firmwide_greeks_df.loc[i, delta_column]
                            past_value = agg_firmwide_greeks_df[agg_firmwide_greeks_df['Date'] == last_date][delta_column].values[0]
                            agg_firmwide_greeks_df.loc[i, wow_delta_column] = current_value - past_value

        return agg_firmwide_greeks_df

    def csv_setup(self):

    # STEP 0: I guess step 0 is now getting the MSTR position lol

        mstr_time_series_df = self.mstr_df_setup()
        mstr_time_series_df.to_csv("../output/mstr_time_series.csv")
        mstr_time_series_df['Date'] = pd.to_datetime(mstr_time_series_df['Date'])

    # STEP 1: Pull the unedited Novo time series (this will have to have the MSTR spread position subtracted from it)

        novo_time_series_df = self.novo_df_setup(mstr_time_series_df)
        novo_time_series_df.to_csv("../output/novo_time_series.csv")
        novo_time_series_df['Date'] = pd.to_datetime(novo_time_series_df['Date'])

    # STEP 2: Pull the unedited GDLP time series (this will need to have Novo's size subtracted from it)

        gdlp_ex_novo_time_series_df = self.gdlp_ex_novo_df_setup(novo_time_series_df, mstr_time_series_df)
        gdlp_ex_novo_time_series_df.to_csv("../output/gdlp_ex_novo_time_series.csv")
        gdlp_ex_novo_time_series_df['Date'] = pd.to_datetime(gdlp_ex_novo_time_series_df['Date'])

    #STEP 3: Pull derivs' data from databricks and then rename the columns

        # THIS IS NOT PIVOTED
        cleaned_derivs_df = parse_and_clean_gdt_data(self.derivs_csv)

        first_column_name = cleaned_derivs_df.columns[0]
        cleaned_derivs_df = cleaned_derivs_df.rename(columns={first_column_name: 'Date'})

        cleaned_derivs_df = self.rename_derivs_columns(cleaned_derivs_df)
        cleaned_derivs_df.to_csv("../output/derivs_time_series.csv")
        cleaned_derivs_df['Date'] = pd.to_datetime(cleaned_derivs_df['Date'])

        # EARLIEST DATE WE ARE INTERESTED IN
        earliest_date = pd.to_datetime('2024-06-12')

    #STEP 4: Need to combine into aggregate sums

        novo_time_series_df.reset_index(inplace=True)
        gdlp_ex_novo_time_series_df.reset_index(inplace=True)
        cleaned_derivs_df.reset_index(inplace=True)

        common_dates = set(gdlp_ex_novo_time_series_df['Date']).intersection(set(novo_time_series_df['Date'])).intersection(set(cleaned_derivs_df['Date']))
        common_dates = sorted(common_dates)

        agg_firmwide_greeks_df = cleaned_derivs_df[cleaned_derivs_df['Date'].isin(common_dates)].copy()

        # Create initial aggregate delta columns
        for asset in ['BTC', 'ETH', 'SOL', 'Alts', 'Passive Beta']:
            agg_firmwide_greeks_df[f'Aggregate {asset} $delta'] = 0

        multiplied_flag = False
        agg_firmwide_greeks_df = self.update_agg_firmwide_greeks(gdlp_ex_novo_time_series_df, agg_firmwide_greeks_df, multiplied_flag)
        multiplied_flag = True
        agg_firmwide_greeks_df = self.update_agg_firmwide_greeks(novo_time_series_df, agg_firmwide_greeks_df, multiplied_flag)
        agg_firmwide_greeks_df = self.update_agg_firmwide_greeks_with_derivs(cleaned_derivs_df, agg_firmwide_greeks_df)

    #STEP 5: And then append these dfs...

        for group, time_series in [("GDLP ex-Novo", gdlp_ex_novo_time_series_df), ("Novo", novo_time_series_df)]:
            for asset in ['BTC', 'ETH', 'SOL', 'Alts', 'Passive Beta']:
                agg_firmwide_greeks_df = self.append_deltas(group, asset, time_series, agg_firmwide_greeks_df, earliest_date)

        cleaned_derivs_df = cleaned_derivs_df[cleaned_derivs_df['Date'] >= earliest_date]  # Filter by earliest date
        
        for index, row in cleaned_derivs_df.iterrows():
            date = row['Date']
            for asset in ['Alts', 'BTC', 'ETH', 'SOL']:
                delta_column = f"Derivs {asset} $delta"
                if delta_column not in agg_firmwide_greeks_df.columns:
                    agg_firmwide_greeks_df[delta_column] = np.nan
                if date in agg_firmwide_greeks_df['Date'].values:
                    agg_firmwide_greeks_df.loc[agg_firmwide_greeks_df['Date'] == date, delta_column] = row[f"Derivs {asset} $delta"]

    #STEP 6: Now want to add the mining constant to our aggregate passive beta

        agg_firmwide_greeks_df = self.add_btc_price_and_mining_constant(agg_firmwide_greeks_df)

        # and now get the total, now that we have all other deltas filled in

        agg_firmwide_greeks_df = self.add_aggregate_total_delta(agg_firmwide_greeks_df)

    #STEP 7: Now want to add vol and var numbers

        agg_firmwide_greeks_df = self.add_var_and_vol(agg_firmwide_greeks_df)

    # STEP 8: Add week-over-week deltas to aggregate and other 3 groups.

        agg_firmwide_greeks_df = self.calculate_week_over_week_delta(agg_firmwide_greeks_df, ['GDLP ex-Novo', 'Novo', 'Derivs', 'Aggregate'], ['BTC', 'ETH', 'SOL', 'Alts', 'Passive Beta'])

        # And finally add wow delta for aggregate total

        agg_firmwide_greeks_df['Aggregate Total $delta WoW'] = np.nan

        for i in range(len(agg_firmwide_greeks_df)):
            current_date = agg_firmwide_greeks_df.loc[i, 'Date']
            one_week_ago = current_date - pd.Timedelta(days=7)
            
            past_dates = agg_firmwide_greeks_df[agg_firmwide_greeks_df['Date'] <= one_week_ago]['Date']
            if not past_dates.empty:
                last_date = past_dates.max()
                current_value = agg_firmwide_greeks_df.loc[i, 'Aggregate Total $delta']
                past_value = agg_firmwide_greeks_df[agg_firmwide_greeks_df['Date'] == last_date]['Aggregate Total $delta'].values[0]
                agg_firmwide_greeks_df.loc[i, 'Aggregate Total $delta WoW'] = current_value - past_value

        agg_firmwide_greeks_df.to_csv('../output/agg_firmwide_greeks.csv')
        agg_firmwide_greeks_df.to_excel('../output/agg_firmwide_greeks.xlsx')

        return agg_firmwide_greeks_df

    # STEP 3: Create table
        
    def create_snapshot_data(self, data):
        latest_data = data.iloc[-1]

        # Initialize a dictionary to create the new DataFrame
        table_data = {
            'asset': [],
            'delta': [],
            'w-o-w delta': [],
            'delta guideline': [],
            'gamma': [],
            'vega': [],
            'vega guideline': [],
            'vol': [],
            'vol guideline': [],
            'var': [],
            'GDLP ex-Novo delta': [],
            'GDLP ex-Novo delta WoW': [],
            'Novo delta': [],
            'Novo delta WoW': [],
            'Derivatives (GDT) delta': [],
            'Derivatives (GDT) delta WoW': []
        }

        # Define the assets
        assets = ['BTC', 'ETH', 'SOL', 'Other Digital Assets', 'Passive Beta']

        # Manually input the guideline numbers
        delta_guideline_total = 1375  # Example value in $ mm
        vega_guideline_total = 1.5  # Example value in $ mm
        vol_guideline_total = 450  # Example value in $ mm

        # Populate the table_data dictionary
        for asset in assets:
            if asset == 'Other Digital Assets':
                table_data['asset'].append('Other Digital Assets')
                table_data['delta'].append(latest_data.get('Aggregate Alts $delta', 0))
                table_data['w-o-w delta'].append(latest_data.get('Aggregate Alts $delta WoW', 0))
                table_data['delta guideline'].append(None)
                table_data['gamma'].append(latest_data.get('Derivs Alts $gamma', 0))
                table_data['vega'].append(latest_data.get('Derivs Alts $vega', 0))
                table_data['vega guideline'].append(None)
                table_data['vol'].append(None)
                table_data['vol guideline'].append(None)
                table_data['var'].append(None)
                table_data['GDLP ex-Novo delta'].append(latest_data.get('GDLP ex-Novo Alts $delta', 0))
                table_data['GDLP ex-Novo delta WoW'].append(latest_data.get('GDLP ex-Novo Alts $delta WoW', 0))
                table_data['Novo delta'].append(latest_data.get('Novo Alts $delta', 0))
                table_data['Novo delta WoW'].append(latest_data.get('Novo Alts $delta WoW', 0))
                table_data['Derivatives (GDT) delta'].append(latest_data.get('Derivs Alts $delta', 0))
                table_data['Derivatives (GDT) delta WoW'].append(latest_data.get('Derivs Alts $delta WoW', 0))
            elif asset == 'Passive Beta':
                table_data['asset'].append('Passive Beta')
                table_data['delta'].append(latest_data.get('Aggregate Passive Beta $delta', 0))
                table_data['w-o-w delta'].append(latest_data.get('Aggregate Passive Beta $delta WoW', 0))
                table_data['delta guideline'].append(None)
                table_data['gamma'].append(None)
                table_data['vega'].append(None)
                table_data['vega guideline'].append(None)
                table_data['vol'].append(None)
                table_data['vol guideline'].append(None)
                table_data['var'].append(None)
                table_data['GDLP ex-Novo delta'].append(latest_data.get('GDLP ex-Novo Passive Beta $delta', 0))
                table_data['GDLP ex-Novo delta WoW'].append(latest_data.get('GDLP ex-Novo Passive Beta $delta WoW', 0))
                table_data['Novo delta'].append(latest_data.get('Novo Passive Beta $delta', 0))
                table_data['Novo delta WoW'].append(latest_data.get('Novo Passive Beta $delta WoW', 0))
                table_data['Derivatives (GDT) delta'].append(latest_data.get('Derivs Passive Beta $delta', 0))
                table_data['Derivatives (GDT) delta WoW'].append(latest_data.get('Derivs Passive Beta $delta WoW', 0))
            else:
                table_data['asset'].append(asset)
                table_data['delta'].append(latest_data.get(f'Aggregate {asset} $delta', 0))
                table_data['w-o-w delta'].append(latest_data.get(f'Aggregate {asset} $delta WoW', 0))
                table_data['delta guideline'].append(None)
                table_data['gamma'].append(latest_data.get(f'Derivs {asset} $gamma', 0))
                table_data['vega'].append(latest_data.get(f'Derivs {asset} $vega', 0))
                table_data['vega guideline'].append(None)
                table_data['vol'].append(latest_data.get(f'{asset} 6m Vol', 0))
                table_data['vol guideline'].append(None)
                table_data['var'].append(latest_data.get(f'{asset} VaR95', 0))
                table_data['GDLP ex-Novo delta'].append(latest_data.get(f'GDLP ex-Novo {asset} $delta', 0))
                table_data['GDLP ex-Novo delta WoW'].append(latest_data.get(f'GDLP ex-Novo {asset} $delta WoW', 0))
                table_data['Novo delta'].append(latest_data.get(f'Novo {asset} $delta', 0))
                table_data['Novo delta WoW'].append(latest_data.get(f'Novo {asset} $delta WoW', 0))
                table_data['Derivatives (GDT) delta'].append(latest_data.get(f'Derivs {asset} $delta', 0))
                table_data['Derivatives (GDT) delta WoW'].append(latest_data.get(f'Derivs {asset} $delta WoW', 0))

        # Add total row with manual guideline values
        table_data['asset'].append('Total')
        table_data['delta'].append(sum(table_data['delta']))
        table_data['w-o-w delta'].append(sum(table_data['w-o-w delta']))
        table_data['delta guideline'].append(delta_guideline_total)
        table_data['gamma'].append(sum([val for val in table_data['gamma'] if val is not None]))
        table_data['vega'].append(sum([val for val in table_data['vega'] if val is not None]))
        table_data['vega guideline'].append(vega_guideline_total)
        table_data['vol'].append(sum([val for val in table_data['vol'] if val is not None]))
        table_data['vol guideline'].append(vol_guideline_total)
        table_data['var'].append(sum([val for val in table_data['var'] if val is not None]))
        table_data['GDLP ex-Novo delta'].append(sum(table_data['GDLP ex-Novo delta']))
        table_data['GDLP ex-Novo delta WoW'].append(sum(table_data['GDLP ex-Novo delta WoW']))
        table_data['Novo delta'].append(sum(table_data['Novo delta']))
        table_data['Novo delta WoW'].append(sum(table_data['Novo delta WoW']))
        table_data['Derivatives (GDT) delta'].append(sum(table_data['Derivatives (GDT) delta']))
        table_data['Derivatives (GDT) delta WoW'].append(sum(table_data['Derivatives (GDT) delta WoW']))

        table_df = pd.DataFrame(table_data)
        table_df.set_index('asset', inplace=True)
        table_df_millions = table_df.copy()

        table_df_millions[['delta', 'w-o-w delta', 'delta guideline', 'gamma', 'vega', 'vega guideline', 'vol', 'vol guideline', 'var', 'GDLP ex-Novo delta', 'GDLP ex-Novo delta WoW', 'Novo delta', 'Novo delta WoW', 'Derivatives (GDT) delta', 'Derivatives (GDT) delta WoW']] = table_df_millions[['delta', 'w-o-w delta', 'delta guideline', 'gamma', 'vega', 'vega guideline', 'vol', 'vol guideline', 'var', 'GDLP ex-Novo delta', 'GDLP ex-Novo delta WoW', 'Novo delta', 'Novo delta WoW', 'Derivatives (GDT) delta', 'Derivatives (GDT) delta WoW']].apply(lambda x: x / 1e6)
        
        return table_df_millions

    def create_firmwide_table(self, data, latest_date):
        df = data

        delta_guideline_total = 1375  # Example value in $ mm
        vega_guideline_total = 1.5  # Example value in $ mm
        vol_guideline_total = 450  # Example value in $ mm

        df.loc['Total', 'delta guideline'] = delta_guideline_total
        df.loc['Total', 'vega guideline'] = vega_guideline_total
        df.loc['Total', 'vol guideline'] = vol_guideline_total 

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[
                    'Asset', 
                    'Delta ($ mm)', 
                    'WoW Delta ($ mm)', 
                    'Delta Guideline ($ mm)', 
                    'Gamma ($ mm)', 
                    'Vega ($ mm)', 
                    'Vega Guideline ($ mm)', 
                    'Vol ($ mm)', 
                    'Vol Guideline ($ mm)', 
                    'VaR 95 ($ mm)'
                ],
                fill_color=COLORS['Galaxy Gray 5'],
                align='center',
                font=dict(family="Forma DJR Display", size=16, color=COLORS['Galaxy White'])
            ),
            cells=dict(
                values=[
                    df.index,
                    df['delta'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['w-o-w delta'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['delta guideline'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['gamma'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['vega'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['vega guideline'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['vol'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['vol guideline'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['var'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                ],
                fill_color='white',
                align='center',
                font=dict(family="Forma DJR Display", size=18, color=COLORS['Galaxy Black']),
                height=35
            )
        )])

        fig.update_layout(
            title=f"Firmwide Greek Exposures - {latest_date}",
            title_font=dict(size=24, family="Forma DJR Display", color=COLORS['Galaxy Black']),
            autosize=False,
            width=1800,
            height=400,
            margin=dict(l=20, r=20, t=80, b=0)
        )

        fig.write_image('../output/table.png', scale=3)

        fig.show()

        return fig

    def create_split_table(self, data, latest_date):
        df = data

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[
                    'Asset',
                    'GDLP ex-Novo Delta ($ mm)',
                    'GDLP ex-Novo Delta WoW ($ mm)',
                    'Novo Delta ($ mm)',
                    'Novo Delta WoW ($ mm)',
                    'Derivatives (GDT) Delta ($ mm)',
                    'Derivatives (GDT) Delta WoW ($ mm)'
                ],
                fill_color=COLORS['Galaxy Gray 5'],
                align='center',
                font=dict(family="Forma DJR Display", size=16, color=COLORS['Galaxy White'])
            ),
            cells=dict(
                values=[
                    df.index,
                    df['GDLP ex-Novo delta'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['GDLP ex-Novo delta WoW'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['Novo delta'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['Novo delta WoW'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['Derivatives (GDT) delta'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else ""),
                    df['Derivatives (GDT) delta WoW'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                ],
                fill_color='white',
                align='center',
                font=dict(family="Forma DJR Display", size=18, color=COLORS['Galaxy Black']),
                height=35
            )
        )])

        fig.update_layout(
            title=f"Group-wise Delta Exposures - {latest_date}",
            title_font=dict(size=24, family="Forma DJR Display", color=COLORS['Galaxy Black']),
            autosize=False,
            width=2000,
            height=400, 
            margin=dict(l=20, r=20, t=80, b=0)
        )

        fig.write_image('../output/split_table.png', scale=3)

        fig.show()
        
        return fig
    
    def create_pie_chart(self, snapshot_data, latest_date):
        # Ensure 'Asset' is a column
        df = snapshot_data.reset_index()
        # Drop the 'Total' row and reset the index
        df = df.drop([5])
        # Rename the index column to 'Asset'
        df = df.rename(columns={'index': 'Asset'})
        
        # Create the pie chart
        fig = px.pie(df, names='asset', values='delta', color='asset', color_discrete_map=asset_colors)
        fig.update_traces(textinfo='percent', textposition='outside')

        # Update layout
        fig.update_layout(
            height=400,
            title_text=f'Firmwide Delta, By Asset - {latest_date}',
            showlegend=True,
            font=dict(family="Forma DJR Display", size=20, color=COLORS["Galaxy Black"]),
            margin=dict(l=30, r=30, t=70, b=30)  # Adjust bottom margin for more space
        )

        # Save the pie chart as an image
        fig.write_image('../output/pie_chart.png')

        # Display the pie chart
        fig.show()
        return fig

    # STEP 4: Create time series charts

    def create_delta_time_series_plot(self, data, latest_date):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        delta_assets = ['BTC', 'ETH', 'SOL', 'Passive Beta', 'Alts']
        for asset in delta_assets:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[f'Aggregate {asset} $delta'], mode='lines', name=asset,
                        line=dict(color=asset_colors[asset]), showlegend=True),
                secondary_y=False
            )

        # Adding Bitcoin price on the right y-axis
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BTC Reference Price T-0'], mode='lines', name='BTC Price',
                    line=dict(color='blue', dash='dot'), showlegend=True),
            secondary_y=True
        )

        fig.update_layout(
            height=600,
            title_text=f"Firmwide Delta Time Series - {latest_date}",
            showlegend=True,
            font=dict(family="Forma DJR Display", size=12, color="black"),
        )

        # Update x-axis and y-axis
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Delta ($)', tickprefix="$", tickformat=",.2s", secondary_y=False)
        fig.update_yaxes(title_text='BTC Price', tickprefix="$", tickformat=",.2s", secondary_y=True)

        fig.update_yaxes(tickprefix="$", tickformat=",.2s")

        fig.update_traces(
            hovertemplate='%{y:$,.0f}',
        )

        fig.write_image('../output/delta_time_series.png')

        fig.show()
        return fig
    
    def create_vega_time_series_plot(self, data, latest_date):
        fig = go.Figure()

        vega_assets = ['BTC', 'ETH', 'SOL', 'Alts']
        for asset in vega_assets:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[f'Derivs {asset} $vega'], mode='lines', name=asset,
                        line=dict(color=asset_colors[asset]), showlegend=True),
            )

        fig.update_layout(
            height=600,
            title_text=f"Firmwide Vega Time Series - {latest_date}",
            showlegend=True,
            font=dict(family="Forma DJR Display", size=12, color="black"),
        )

        # Update x-axis and y-axis
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Vega ($)', tickprefix="$", tickformat=",.2s")

        fig.update_traces(
            hovertemplate='%{y:$,.0f}',
        )

        fig.write_image('../output/vega_time_series.png')

        fig.show()
        return fig

    # Function to format numbers
    def format_number(self, value):
        if pd.isna(value):
            return ""
        if abs(value) >= 1e6:
            return f"${int(value / 1e6)}mm"
        elif abs(value) >= 1e3:
            return f"${int(value / 1e3)}k"
        else:
            return f"${int(value)}"

    def populate_shock_matrix(self, data, asset, shock_type):
        df = data
        shock_columns = [col for col in df.columns if col.startswith(f'{asset} {shock_type} Shock')]

        # Why is this -2, shouldn't it be -1? Or rather I'm going to change it to -1 now, it was -2 before
        shock_data = df[shock_columns].iloc[-1]

        shock_matrix = pd.DataFrame(index=[-25, -10, -5, 0, 5, 10, 25], columns = [-25, -10, -5, 0, 5, 10, 25])
        for col in shock_columns:
            try:
                spot_part = col.split('Spot')[1].split(',')[0].strip()
                vol_part = col.split('Vol')[1].strip()

                spot_shock = int(spot_part)
                vol_shock = int(vol_part)
                
                shock_matrix.loc[vol_shock, spot_shock] = shock_data[col]
            except ValueError as e:
                print(f'Error processing column {col}: {e}')

        shock_matrix = shock_matrix.apply(pd.to_numeric, errors='coerce')
        
        return shock_matrix

    def create_shock_matrices(self, data, assets, latest_date):
        df = data
        figures = {}
        
        for asset in assets:
            delta_shock_matrix = self.populate_shock_matrix(df, asset, 'Delta')
            pnl_shock_matrix = self.populate_shock_matrix(df, asset, 'PnL')

            delta_shock_matrix = delta_shock_matrix.fillna(0)
            pnl_shock_matrix = pnl_shock_matrix.fillna(0)

            # Create a meshgrid for consistent spacing
            x = np.linspace(-25, 25, 7)
            y = np.linspace(-25, 25, 7)
            x_ticks = ['-25', '-10', '-5', '0', '5', '10', '25']
            y_ticks = ['-25', '-10', '-5', '0', '5', '10', '25']
            
            # Format text for cells
            delta_text = np.vectorize(self.format_number)(delta_shock_matrix.values)
            pnl_text = np.vectorize(self.format_number)(pnl_shock_matrix.values)

            # Plot Delta Shock Matrix
            fig_delta = go.Figure(data=go.Heatmap(
                z=delta_shock_matrix.values,
                x=x,
                y=y,
                colorscale='RdYlGn',
                text=delta_text,
                hoverinfo='text',
                showscale=False  # Hide color scale
            ))
            
            fig_delta.update_layout(
                title=f'Firmwide {asset} Delta Shock Matrix - {latest_date}',
                font=dict(family="Forma DJR Display", size=18, color=COLORS["Galaxy Black"]),
                xaxis=dict(
                    title="Spot Shock",
                    tickmode='array',
                    tickvals=x,
                    ticktext=x_ticks
                ),
                yaxis=dict(
                    title="Vol Shock",
                    tickmode='array',
                    tickvals=y,
                    ticktext=y_ticks
                ),
                margin=dict(l=100, r=100, t=100, b=100)  # Adjust bottom margin for more space
            )

            fig_delta.update_traces(
                texttemplate='%{text}',
                textfont={'size': 12}
            )

            fig_delta.write_image(f'../output/{asset}_delta_shock_matrix.png')
            figures[f'{asset.lower()}_delta_shock_matrix'] = fig_delta  # Add the figure to the dictionary
            
            fig_delta.show()
            
            # Plot PnL Shock Matrix
            fig_pnl = go.Figure(data=go.Heatmap(
                z=pnl_shock_matrix.values,
                x=x,
                y=y,
                colorscale='RdYlGn',
                text=pnl_text,
                hoverinfo='text',
                showscale=False  # Hide color scale
            ))
            
            fig_pnl.update_layout(
                title=f'Firmwide {asset} PnL Shock Matrix - {latest_date}',
                font=dict(family="Forma DJR Display", size=18, color=COLORS["Galaxy Black"]),
                xaxis=dict(
                    title="Spot Shock",
                    tickmode='array',
                    tickvals=x,
                    ticktext=x_ticks
                ),
                yaxis=dict(
                    title="Vol Shock",
                    tickmode='array',
                    tickvals=y,
                    ticktext=y_ticks
                ),
                margin=dict(l=100, r=100, t=100, b=100)  # Adjust bottom margin for more space
            )

            fig_pnl.update_traces(
                texttemplate='%{text}',
                textfont={'size': 12}
            )

            fig_pnl.write_image(f'../output/{asset}_pnl_shock_matrix.png')
            figures[f'{asset.lower()}_pnl_shock_matrix'] = fig_pnl  # Add the figure to the dictionary

            fig_pnl.show()

        return figures

    def run_firmwide_risk_report(self):
        data = self.csv_setup()
        #data = pd.read_csv("../output/agg_firmwide_greeks.csv")
        latest_date = pd.to_datetime(data['Date']).iloc[-1].strftime('%Y-%m-%d')
        data.set_index('Date', inplace=True)
        
        snapshot_data = self.create_snapshot_data(data)
        
        table_fig = self.create_firmwide_table(snapshot_data, latest_date)
        split_table_fig = self.create_split_table(snapshot_data, latest_date)

        pie_chart_fig = self.create_pie_chart(snapshot_data, latest_date)

        delta_time_series_fig = self.create_delta_time_series_plot(data, latest_date)
        vega_time_series_fig = self.create_vega_time_series_plot(data, latest_date)
        

        assets = ['Alts', 'BTC', 'ETH', 'SOL']
        selected_greeks = ['$delta', '$vega']
        shock_matrix_figs = self.create_shock_matrices(data, ['BTC', 'ETH', 'SOL'], latest_date)
        
        return {
            'latest_date' : latest_date,
            'table': table_fig,
            'split_table': split_table_fig,
            'pie_chart': pie_chart_fig,
            'delta_time_series': delta_time_series_fig,
            'vega_time_series': vega_time_series_fig,
            **shock_matrix_figs
        }
