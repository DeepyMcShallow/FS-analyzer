# data_loader.py
import pandas as pd
from datetime import datetime
import io # Required for reading uploaded file object

# --- Constants for Holdings File Parsing ---
HOLDINGS_HEADER_ROW = 1
HOLDINGS_TABLE_1_END_ROW_TEXT = "Totals" 
COL_ASSET_CLASS_NORM = "asset class"
COL_NAME_OF_INSTITUTION_NORM = "name of institution"
COL_NAME_OF_INVESTMENT_ITEM_NORM = "name / kind of investment item" 
COL_INT_FIXED_INC_NORM = "internally held fixed income investments"
COL_EXT_FIXED_INC_NORM = "externally held fixed income investments"
COL_LISTED_EQUITIES_NORM = "listed equities"
COL_CURRENCY_NORM = "currency"
COL_SECURITY_ID_NORM = "security identifier"
COL_WEIGHTING_NORM = "weighting(%)"

EXPECTED_HOLDING_DF_COLUMNS = ["CanonicalHoldingName", "Ticker", "AssetClass", "Weighting", "Currency"]


def normalize_column_names(df):
    """Converts column names to lowercase and strips whitespace."""
    df.columns = [str(col).lower().strip().replace("\n", " ").replace("\r", " ").replace("  ", " ") for col in df.columns]
    return df

def load_unit_price_history(file_input):
    """
    Loads the unit price history.
    file_input can be a file path (string) or an uploaded file object (BytesIO/StringIO).
    """
    try:
        if isinstance(file_input, str): # It's a file path
            file_path = file_input.replace("\\", "/")
            df = pd.read_csv(file_path)
            current_file_name = file_path
        elif hasattr(file_input, 'getvalue'): # It's an uploaded file object (like BytesIO)
            # For BytesIO, pandas can read it directly
            df = pd.read_csv(file_input)
            current_file_name = getattr(file_input, 'name', 'Uploaded File')
        else:
            print("Error: Invalid file input type for unit price history.")
            return pd.DataFrame(columns=['Date', 'FundName', 'UnitPrice'])

        original_first_col_name = df.columns[0]
        df = df.rename(columns={original_first_col_name: str(original_first_col_name).lower().strip()})
        if df.columns[0] != 'date':
            print(f"Warning: Unit Price History ({current_file_name}) - Expected 'date' as first column, found '{df.columns[0]}'.")
        df = df.rename(columns={df.columns[0]: 'Date'}) 
        df_long = pd.melt(df, id_vars=['Date'], var_name='FundName', value_name='UnitPrice')
        try:
            df_long['Date'] = pd.to_datetime(df_long['Date'], dayfirst=True, errors='coerce')
        except Exception:
            df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')
        df_long.dropna(subset=['UnitPrice', 'Date'], inplace=True)
        
        print(f"Successfully loaded and transformed unit price history from {current_file_name}.")
        if not df_long.empty:
            print(f"Found data for {df_long['FundName'].nunique()} funds from {df_long['Date'].min().strftime('%Y-%m-%d')} to {df_long['Date'].max().strftime('%Y-%m-%d')}.")
        else:
            print(f"Warning: No valid unit price data found after processing {current_file_name}.")
        return df_long
    except FileNotFoundError: # Only relevant if file_input was a path
        print(f"Error: Unit price history file not found at {file_input}")
        return pd.DataFrame(columns=['Date', 'FundName', 'UnitPrice'])
    except Exception as e:
        print(f"Error loading unit price history from {getattr(file_input, 'name', str(file_input))}: {e}")
        return pd.DataFrame(columns=['Date', 'FundName', 'UnitPrice'])


def load_fund_holdings(file_path="data/Accumulation-High-Growth.csv"):
    """Loads fund holdings, focusing on Table 1."""
    encodings_to_try = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    df = None
    file_path = file_path.replace("\\", "/")
    # print(f"\n--- Processing Holdings File: {file_path} ---") # Already in CLI

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, header=HOLDINGS_HEADER_ROW, encoding=encoding, skipinitialspace=True)
            df = normalize_column_names(df)
            # print(f"Successfully read {file_path} with encoding {encoding}.") # Already in CLI
            # print(f"Available columns after normalization: {df.columns.tolist()}") 
            break 
        except UnicodeDecodeError: pass
        except FileNotFoundError:
            print(f"Error: Holdings file not found at {file_path}")
            return pd.DataFrame(columns=EXPECTED_HOLDING_DF_COLUMNS)
        except Exception as e: print(f"Error reading {file_path} with {encoding}: {e}")

    if df is None:
        print(f"Error: Could not read holdings file {file_path}.")
        return pd.DataFrame(columns=EXPECTED_HOLDING_DF_COLUMNS)

    try:
        first_col_table1_norm = df.columns[0] 
        if COL_ASSET_CLASS_NORM in df.columns:
            first_col_table1_norm = COL_ASSET_CLASS_NORM
        else:
            print(f"Warning: Column '{COL_ASSET_CLASS_NORM}' not found in {file_path}. Using '{first_col_table1_norm}' for Table 1 end detection.")
        
        end_table_1_idx = df[df[first_col_table1_norm].astype(str).str.contains(HOLDINGS_TABLE_1_END_ROW_TEXT, case=False, na=False)].index.min()
        
        if pd.isna(end_table_1_idx):
            key_cols_for_na_check_norm = [col for col in [COL_NAME_OF_INSTITUTION_NORM, COL_SECURITY_ID_NORM, COL_WEIGHTING_NORM] if col in df.columns]
            if key_cols_for_na_check_norm:
                 all_nan_rows = df[df[key_cols_for_na_check_norm].isnull().all(axis=1)]
                 if not all_nan_rows.empty: end_table_1_idx = all_nan_rows.index.min()

        if not pd.isna(end_table_1_idx):
            df_table1 = df.iloc[:end_table_1_idx].copy()
            # print(f"Table 1 in {file_path} identified to end at or before row index {end_table_1_idx}.") # Already in CLI
        else:
            df_table1 = df.dropna(how='all').copy() 
            # print(f"Warning: Could not definitively find end of Table 1 in {file_path}. Using all non-empty rows.") # Already in CLI
        
        processed_holdings = []
        for idx, row in df_table1.iterrows():
            asset_class = str(row.get(COL_ASSET_CLASS_NORM, "")).strip()
            name_institution = str(row.get(COL_NAME_OF_INSTITUTION_NORM, "")).strip()
            name_investment_item = str(row.get(COL_NAME_OF_INVESTMENT_ITEM_NORM, "")).strip()
            name_int_fixed_inc = str(row.get(COL_INT_FIXED_INC_NORM, "")).strip()
            name_ext_fixed_inc = str(row.get(COL_EXT_FIXED_INC_NORM, "")).strip()
            name_listed_equity = str(row.get(COL_LISTED_EQUITIES_NORM, "")).strip()
            currency = str(row.get(COL_CURRENCY_NORM, "AUD")).strip()
            security_id = str(row.get(COL_SECURITY_ID_NORM, "")).strip()
            
            weighting_val_from_csv = row.get(COL_WEIGHTING_NORM, "0") 
            try:
                weighting_str = str(weighting_val_from_csv).strip().rstrip('%')
                if weighting_str and weighting_str not in ['nan', '-', ''] and not pd.isna(pd.to_numeric(weighting_str, errors='coerce')):
                    weighting = float(weighting_str)
                else:
                    weighting = 0.0
            except ValueError:
                weighting = 0.0
            
            canonical_name = "Unspecified Holding"
            if name_listed_equity and name_listed_equity.lower() != 'nan':
                canonical_name = name_listed_equity
            elif name_investment_item and name_investment_item.lower() != 'nan': 
                canonical_name = name_investment_item
            elif "fixed income" in asset_class.lower() or "bond" in asset_class.lower():
                if name_int_fixed_inc and name_int_fixed_inc.lower() != 'nan': canonical_name = name_int_fixed_inc
                elif name_ext_fixed_inc and name_ext_fixed_inc.lower() != 'nan': canonical_name = name_ext_fixed_inc
                elif name_institution and name_institution.lower() != 'nan': canonical_name = name_institution
                elif security_id and security_id.lower() != 'nan': canonical_name = security_id 
            elif name_institution and name_institution.lower() != 'nan':
                canonical_name = name_institution
            elif security_id and security_id.lower() != 'nan': 
                canonical_name = security_id
            
            ticker_clean = security_id if security_id.lower() != 'nan' else ""
            is_meaningful_name = canonical_name.lower() not in ['unspecified holding', 'nan', '']
            is_meaningful_ticker = ticker_clean != ""
            
            if is_meaningful_name or is_meaningful_ticker or weighting != 0.0:
                processed_holdings.append({
                    "CanonicalHoldingName": canonical_name,
                    "Ticker": ticker_clean,
                    "AssetClass": asset_class if asset_class.lower() != 'nan' else "Unclassified",
                    "Weighting": weighting,
                    "Currency": currency if currency.lower() != 'nan' else "AUD"
                })
        
        if not processed_holdings:
            # print(f"Warning: No holdings were processed from Table 1 in {file_path}.") # Already in CLI
            return pd.DataFrame(columns=EXPECTED_HOLDING_DF_COLUMNS)

        holdings_df = pd.DataFrame(processed_holdings)
        holdings_df = holdings_df[~(
            (holdings_df['CanonicalHoldingName'].str.lower().isin(['unspecified holding', 'nan', ''])) &
            (holdings_df['Ticker'] == "") &
            (holdings_df['Weighting'] == 0.0)
        )]
        # print(f"Successfully loaded {len(holdings_df)} direct holdings from Table 1 in {file_path}.") # Already in CLI
        return holdings_df

    except Exception as e:
        print(f"Critical error processing holdings from {file_path} after successful read: {e}")
        return pd.DataFrame(columns=EXPECTED_HOLDING_DF_COLUMNS)

# (Keep the if __name__ == '__main__': block from v5 for testing data_loader.py if desired)
