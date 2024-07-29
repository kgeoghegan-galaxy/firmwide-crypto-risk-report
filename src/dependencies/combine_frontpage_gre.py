from dependencies.construct_frontpage_time_series import ConstructFrontpageTimeSeries
from dependencies.handle_gre_data import GREDataProcessor
import datetime
import re
import pandas as pd

class CombineFrontpageGRE:
    def __init__(self, base_directory, gre_data_filepath):
        self.base_directory = base_directory
        self.gre_data_filepath = gre_data_filepath
    
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
        
    def pull_gre_data(self):
        processor = GREDataProcessor(self.gre_data_filepath)
        gre_df = processor.process()
        return gre_df

    def pull_frontpage_data(self, date_str):
        novo_filename_pattern = f"Frontpage - (Novo) - {date_str.replace('-', '')}"
        ticker_groups = {'BTC': ['BTC']}
        metrics = ['Mkt Val (ex DLOM) T-0 ($)']

        analysis = ConstructFrontpageTimeSeries(self.base_directory)
        frontpage_time_series = analysis.collect_data(novo_filename_pattern, ticker_groups, metrics, preprocess_tickers=True)
        frontpage_time_series.to_csv('../output/frontpage_time_series.csv', index=False)

        return frontpage_time_series

    def pull_btc_price_data(self, date_str):
        novo_filename_pattern = f"Frontpage - (Novo) - {date_str.replace('-', '')}"
        ticker_groups = {'BTC': ['BTC']}
        metrics = ['Reference Price\n($) T-0']

        analysis = ConstructFrontpageTimeSeries(self.base_directory)
        btc_price_time_series = analysis.collect_data(novo_filename_pattern, ticker_groups, metrics, preprocess_tickers=False)
        btc_price_time_series.to_csv('../output/btc_price_time_series.csv', index=False)

        return btc_price_time_series

    def extract_values(self, frontpage_df, btc_price_df, date_str):
        # Ensure the date and column names match
        print(f"Contents of frontpage_df:\n{frontpage_df}")
        print(f"Contents of btc_price_df:\n{btc_price_df}")
        
        if 'Date' not in frontpage_df.columns or 'Ticker' not in frontpage_df.columns or 'Value' not in frontpage_df.columns:
            raise ValueError("The expected columns are not present in the frontpage DataFrame.")
        
        if 'Date' not in btc_price_df.columns or 'Ticker' not in btc_price_df.columns or 'Value' not in btc_price_df.columns:
            raise ValueError("The expected columns are not present in the BTC price DataFrame.")
        
        # Convert the Date column to string type and clean up any whitespace
        frontpage_df['Date'] = frontpage_df['Date'].astype(str).str.strip()
        btc_price_df['Date'] = btc_price_df['Date'].astype(str).str.strip()
        
        # Filter the frontpage DataFrame for the specific date and metric
        filtered_frontpage_df = frontpage_df[(frontpage_df['Date'] == date_str) & (frontpage_df['Metric'] == 'Mkt Val (ex DLOM) T-0 ($)')]
        print(f"Filtered frontpage_df for date {date_str}:\n{filtered_frontpage_df}")
        
        if filtered_frontpage_df.empty:
            raise ValueError(f"No data found for date {date_str} in frontpage DataFrame.")
        
        novo_alts_value = filtered_frontpage_df[filtered_frontpage_df['Ticker'] == 'Alts']
        if novo_alts_value.empty:
            raise ValueError(f"No 'Alts' ticker found for date {date_str} in frontpage DataFrame.")
        novo_alts_value = novo_alts_value['Value'].values[0]
        
        passive_beta_value = filtered_frontpage_df[filtered_frontpage_df['Ticker'] == 'Passive Beta']
        if passive_beta_value.empty:
            raise ValueError(f"No 'Passive Beta' ticker found for date {date_str} in frontpage DataFrame.")
        passive_beta_value = passive_beta_value['Value'].values[0]
        
        # Filter the BTC price DataFrame for the specific date and metric
        filtered_btc_price_df = btc_price_df[(btc_price_df['Date'] == date_str) & (btc_price_df['Metric'] == 'Reference Price\n($) T-0')]
        print(f"Filtered btc_price_df for date {date_str}:\n{filtered_btc_price_df}")
        
        if filtered_btc_price_df.empty:
            raise ValueError(f"No data found for date {date_str} in BTC price DataFrame.")
        btc_price = filtered_btc_price_df['Value'].values[0]
        
        # Calculate modified 'Passive Beta Delta'
        modified_passive_beta_value = passive_beta_value + (btc_price * 270)
        
        return {
            'Novo Alts Delta': novo_alts_value,
            'Passive Beta Delta': modified_passive_beta_value
        }
        
    def integrate_values(self, gre_df, values_dict):
        # Append the Passive Beta Delta
        gre_df['Passive Beta Delta'] = values_dict['Passive Beta Delta']
        
        # Add the Novo Alts value to the Novo Alts Delta column if it exists, or create it
        if 'Novo Alts Delta' in gre_df.columns:
            gre_df['Novo Alts Delta'] = values_dict['Novo Alts Delta']
        else:
            gre_df = gre_df.assign(**{'Novo Alts Delta': values_dict['Novo Alts Delta']})
        
        # Add Passive Beta Delta to GDLP Total Delta
        if 'GDLP Total Delta' in gre_df.columns:
            gre_df['GDLP Total Delta'] += values_dict['Passive Beta Delta']
        else:
            gre_df = gre_df.assign(**{'GDLP Total Delta': values_dict['Passive Beta Delta']})
        
        return gre_df

    
    def save_combined_data(self, final_df):
        final_df.to_csv('../output/23072024_with_novo.csv', index=False)
        final_df.to_excel('../output/23072024_with_novo.xlsx', index=False)

    def run(self):
        date_str = self.get_date_from_filepath(self.gre_data_filepath)
        
        # Pull GRE data
        gre_df = self.pull_gre_data()
        
        # Pull frontpage data
        frontpage_df = self.pull_frontpage_data(date_str)
        
        # Pull BTC price data
        btc_price_df = self.pull_btc_price_data(date_str)
        
        # Extract values
        values_dict = self.extract_values(frontpage_df, btc_price_df, date_str)
        
        # Integrate values into GRE data without modifying existing data
        final_df = self.integrate_values(gre_df, values_dict)
        
        # Save combined data
        self.save_combined_data(final_df)