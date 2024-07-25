import pandas as pd

class HandleGREData:
    def __init__(self, config):
        self.traders = config['traders']
        self.strategies = config['strategies']

    @staticmethod
    def read_csv_with_unique_header(filepath):
        """
        Reads a CSV file and ensures unique header names.
        
        Parameters:
            filepath (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: DataFrame with unique headers.
        """
        header_df = pd.read_csv(filepath, nrows=2, header=None)
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

        df = pd.read_csv(filepath, skiprows=2, header=None, names=unique_header)
        return df

    @staticmethod
    def combine_columns(df):
        """
        Combines specified columns into a single 'combined' column.
        
        Parameters:
            df (pd.DataFrame): DataFrame to process.
            
        Returns:
            pd.DataFrame: DataFrame with a new 'combined' column.
        """
        df['combined'] = df[['Pod(L2)', 'Book(L3)', 'Underlier', 'Desk Strategy', 'Ticker']].astype(str).apply(
            lambda x: ' '.join(filter(lambda y: y != 'nan' and y != '', x)), axis=1)
        return df

    @staticmethod
    def pivot_dataframe(df):
        """
        Pivots the DataFrame to restructure data based on 'combined' columns.
        
        Parameters:
            df (pd.DataFrame): DataFrame to pivot.
            
        Returns:
            pd.DataFrame: Pivoted DataFrame.
        """
        pivot_df = df.pivot_table(index=None, columns='combined', values='Delta', aggfunc='first').reset_index(drop=True)
        pivot_df.ffill(inplace=True)
        return pivot_df

    @staticmethod
    def sum_deltas(pivot_df, pattern):
        """
        Sums the delta values in the DataFrame that match a given pattern.
        
        Parameters:
            pivot_df (pd.DataFrame): Pivoted DataFrame.
            pattern (str): Pattern to match column names.
            
        Returns:
            float: Sum of the delta values.
        """
        cols = pivot_df.filter(regex=pattern).columns
        return pivot_df[cols].sum().sum() if cols.shape[0] > 0 else 0

    def calculate_trader_deltas(self, pivot_df, trader):
        """
        Calculates the deltas for a specific trader.
        
        Parameters:
            pivot_df (pd.DataFrame): Pivoted DataFrame.
            trader (str): Trader name.
            
        Returns:
            dict: Dictionary of delta values for the trader.
        """
        deltas = {}
        total_sum = 0

        for asset, patterns in self.strategies.items():
            asset_sum = 0
            for pattern in patterns:
                pattern_sum = self.sum_deltas(pivot_df, f'{trader} Crypto {asset} {pattern}')
                asset_sum += pattern_sum
                if pattern_sum != 0:
                    deltas[f'{trader} {asset} {pattern} Delta'] = pattern_sum
            if asset_sum != 0:
                deltas[f'{trader} {asset} Delta'] = asset_sum
            total_sum += asset_sum

        if total_sum != 0:
            deltas[f'{trader} Delta'] = total_sum

        return deltas

    @staticmethod
    def calculate_novo_deltas(pivot_df):
        """
        Calculates the deltas for Novo.
        
        Parameters:
            pivot_df (pd.DataFrame): Pivoted DataFrame.
            
        Returns:
            dict: Dictionary of delta values for Novo.
        """
        deltas = {}
        assets = ['BTC', 'ETH', 'SOL']
        core_patterns = ['Core', 'ETF $', 'Options $']
        total_sum = 0

        for asset in assets:
            asset_sum = 0
            for pattern in core_patterns:
                pattern_sum = HandleGREData.sum_deltas(pivot_df, f'Novo Crypto {asset} {pattern}')
                asset_sum += pattern_sum
                if pattern_sum != 0:
                    deltas[f'Novo {asset} {pattern} Delta'] = pattern_sum
            if asset_sum != 0:
                deltas[f'Novo {asset} Delta'] = asset_sum
            total_sum += asset_sum

        alts_sum = HandleGREData.sum_deltas(pivot_df, 'Novo Crypto .* Core|Options|RelativeValue')
        if alts_sum != 0:
            deltas['Novo Alts Delta'] = alts_sum
        total_sum += alts_sum

        if total_sum != 0:
            deltas['Novo Delta'] = total_sum

        return deltas

    @staticmethod
    def calculate_summary(deltas, novo_deltas):
        """
        Calculates the summary deltas for GDLP ex-Novo and GDLP including Novo.
        
        Parameters:
            deltas (dict): Dictionary of deltas for traders.
            novo_deltas (dict): Dictionary of deltas for Novo.
            
        Returns:
            dict: Summary dictionary of delta values.
        """
        summary = {
            'GDLP ex-Novo BTC Delta': 0,
            'GDLP ex-Novo ETH Delta': 0,
            'GDLP ex-Novo SOL Delta': 0,
            'GDLP ex-Novo Alts Delta': 0,
            'GDLP ex-Novo Delta': 0,
            'GDLP ex-Novo ex-Alts Delta': 0
        }

        for key, value in deltas.items():
            if 'BTC Delta' in key:
                summary['GDLP ex-Novo BTC Delta'] += value
            elif 'ETH Delta' in key:
                summary['GDLP ex-Novo ETH Delta'] += value
            elif 'SOL Delta' in key:
                summary['GDLP ex-Novo SOL Delta'] += value
            elif 'Alts Delta' in key:
                summary['GDLP ex-Novo Alts Delta'] += value
            summary['GDLP ex-Novo Delta'] += value
            if 'Alts' not in key:
                summary['GDLP ex-Novo ex-Alts Delta'] += value

        gdlp_summary = {
            'GDLP BTC Total Delta': summary['GDLP ex-Novo BTC Delta'] + novo_deltas.get('Novo BTC Delta', 0),
            'GDLP ETH Total Delta': summary['GDLP ex-Novo ETH Delta'] + novo_deltas.get('Novo ETH Delta', 0),
            'GDLP SOL Total Delta': summary['GDLP ex-Novo SOL Delta'] + novo_deltas.get('Novo SOL Delta', 0),
            'GDLP Alts Total Delta': summary['GDLP ex-Novo Alts Delta'] + novo_deltas.get('Novo Alts Delta', 0),
            'GDLP Total Delta': summary['GDLP ex-Novo Delta'] + novo_deltas.get('Novo Delta', 0)
        }

        summary.update(gdlp_summary)
        return summary

    def process_csv(self, filepath):
        """
        Processes the CSV file and calculates deltas for all traders and Novo.
        
        Parameters:
            filepath (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: DataFrame with the calculated delta values.
        """
        df = self.read_csv_with_unique_header(filepath)
        df = self.combine_columns(df)
        pivot_df = self.pivot_dataframe(df)

        result_dict = {}

        for trader in self.traders:
            trader_deltas = self.calculate_trader_deltas(pivot_df, trader)
            result_dict.update(trader_deltas)

        novo_deltas = self.calculate_novo_deltas(pivot_df)
        result_dict.update(novo_deltas)

        summary_dict = self.calculate_summary(result_dict, novo_deltas)
        result_dict.update(summary_dict)

        final_df = pd.DataFrame(result_dict, index=['Values'])

        return final_df
