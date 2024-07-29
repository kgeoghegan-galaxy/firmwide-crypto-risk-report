import os
import pandas as pd
from datetime import datetime
import re

class ConstructFrontpageTimeSeries:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.time_series = pd.DataFrame(columns=['Date', 'Ticker', 'Metric', 'Value'])

    def collect_data(self, filename_pattern, ticker_groups, metrics, preprocess_tickers, start_date=(datetime.strptime('12062024', '%d%m%Y').date())):  
        filename_pattern_regex = self.convert_to_regex_pattern(filename_pattern)

        for root, dirs, files in os.walk(self.base_dir):
            if 'frontpages' in root:
                for file in files:
                    if file.endswith('.xlsx') and self.is_target_file(file, filename_pattern_regex):
                        file_date = self.extract_date_from_filename(file)
                        if file_date and file_date >= start_date:
                            file_path = os.path.join(root, file)
                            date_str = self.extract_date_from_path(root)
                            if date_str:
                                date = datetime.strptime(date_str, '%Y/%m/%d').date()
                                self.process_file(file_path, date, ticker_groups, metrics, preprocess_tickers)

        return self.time_series

    def save_to_csv(self, filepath):
        if not self.time_series.empty:
            self.time_series.to_csv(filepath, index=False)
            #print(f"Time series data saved to '{filepath}'.")
        #else:
            #print("No data found to save.")

    def convert_to_regex_pattern(self, pattern):
        pattern_prefix = pattern.rsplit(" - ", 1)[0] + " - "
        pattern_regex = re.escape(pattern_prefix) + r"\d{8}\.xlsx"
        return pattern_regex

    def clean_headers(self, headers):
        return [header.strip().replace('\n', ' ') for header in headers]

    def is_target_file(self, filename, pattern):
        return re.match(pattern, filename) is not None

    def process_file(self, file_path, date, ticker_groups, metrics, preprocess_tickers):
        df = pd.read_excel(io=file_path, index_col=None, skiprows=6, usecols="A:AG")
        
        if preprocess_tickers:
            ticker_groups = self.preprocess_ticker_groups(df)

        for group_name, tickers in ticker_groups.items():
            group_sum = 0
            for ticker in tickers:
                for metric in metrics:
                    value = self.get_value(df, ticker, metric)
                    if value is not None:
                        group_sum += value
            #if group_sum > 0: used to be here, with the 2 lines below indented 
            self.time_series.loc[len(self.time_series)] = [date, group_name, metrics[0], group_sum]
            #print(f"Date: {date}, Ticker: {ticker_groups}, Metric: {metric[0]}, Value: {group_sum}")

    def extract_date_from_path(self, path):
        try:
            parts = path.split(os.sep)
            if len(parts) >= 4:
                return f"{parts[-4]}/{parts[-3]}/{parts[-2]}"
        except IndexError:
            return None

    def extract_date_from_filename(self, filename):
        try:
            date_str = filename.split(" - ")[-1].split(".")[0]
            return datetime.strptime(date_str, '%d%m%Y').date()
        except (IndexError, ValueError):
            return None
        
    def get_value(self, df, ticker, metric):
        try:
            value = df.loc[df['Ticker'] == ticker, metric].values[0]
            return value
        except IndexError:
            #print(f"Value for Ticker: {ticker}, Metric: {metric} not found.")
            return None
    
    def print_available_metrics(self, filename_pattern):
        filename_pattern_regex = self.convert_to_regex_pattern(filename_pattern)
        for root, dirs, files in os.walk(self.base_dir):
            if 'frontpages' in root:
                for file in files:
                    if file.endswith('17072024.xlsx') and self.is_target_file(file, filename_pattern_regex):
                        file_path = os.path.join(root, file)
                        print(file_path)
                        df = pd.read_excel(io=file_path, index_col=None, skiprows=6)
                        print("Copy paste the string for your desired metric that gets outputted in terminal. \nHad a cleaner way to do this but caused problems with parsing some metrics. Low priority to fix.")
                        print("---------------------------------------------")
                        print(df.head())
                        return  # Print metrics from the first file and exit
                    
    def preprocess_ticker_groups(self, df):
        btc = ['BTC']
        eth = ['ETH']
        sol = ['SOL']
        alts = []
        passive_beta = []

        start_processing = False

        for ticker in df['Ticker'].unique():
            if ticker == 'Total ETH':
                start_processing = True
                continue
            if ticker == 'Total Other Digital Assets':
                break
            if ticker in ['Total Other Layer 1s', 'Total DeFi']:
                continue
            if start_processing:
                if ticker.endswith('_PB'):
                    passive_beta.append(ticker)
                else:
                    alts.append(ticker)

        passive_beta.append("BTC_PB")
        passive_beta.append("ETH_PB")
        passive_beta.append("SOL_PB")

        ticker_groups = {
            'BTC': btc,
            'ETH': eth,
            'SOL': sol,
            'Alts': alts,
            'Passive Beta': passive_beta
        }

        return ticker_groups