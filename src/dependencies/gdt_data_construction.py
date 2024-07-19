import pandas as pd
from datetime import datetime

def parse_and_clean_gdt_data(greek_calculations_csv):
    # Load Greek calculations CSV
    greek_df = pd.read_csv(greek_calculations_csv)
    
    # Explicitly name the first column as 'Date'
    greek_df.rename(columns={greek_df.columns[0]: 'Date'}, inplace=True)

    # Parse and format dates
    greek_df['Date'] = greek_df['Date'].apply(parse_and_format_dates)
    
    # Remove rows with invalid dates (if any)
    greek_df = greek_df.dropna(subset=['Date'])
    
    # Optionally, perform additional cleaning or transformations
    
    return greek_df

def parse_and_format_dates(date_str):
    # Function to parse datetime strings and return date strings
    try:
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
        formatted_date = parsed_date.strftime('%Y-%m-%d')
        return formatted_date
    except ValueError:
        return None