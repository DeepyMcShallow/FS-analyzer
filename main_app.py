# main_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import io 
import gspread 
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.oauth2.service_account import Credentials 
import json # For local secrets handling

# --- Import your custom modules ---
try:
    from data_loader import load_unit_price_history, load_fund_holdings
    from news_aggregator import aggregate_news, RSS_FEEDS_CONFIG 
    from sentiment_analyzer import get_sentiment, classify_sentiment_strength
    from impact_estimator import calculate_fund_impact
except ImportError as e:
    st.error(f"CRITICAL ERROR: Could not import one or more custom modules: {e}. ")
    st.stop()

# --- Configuration ---
DATA_DIR = "data" 
UNIT_PRICE_DISPLAY_PRECISION = 6 
GOOGLE_SHEET_NAME = "FutureSaverEstimatesLog"  
WORKSHEET_NAME = "Sheet1" 

HOLDINGS_FILES_INFO = [ 
    {"file": "Accumulation-High-Growth.csv", "name": "High Growth"},
    {"file": "Accumulation-High-Growth-Socially-Conscious (1).csv", "name": "High Growth Socially Conscious"},
    {"file": "Accumulation-High-Growth-Indexed.csv", "name": "High Growth Indexed"},
    {"file": "Accumulation-Balanced.csv", "name": "Balanced"},
    {"file": "Accumulation-Balanced-Socially-Conscious.csv", "name": "Balanced Socially Conscious"},
    {"file": "Accumulation-Balanced-Indexed.csv", "name": "Balanced Indexed"},
    {"file": "Accumulation-Conservative-Balanced.csv", "name": "Conservative Balanced"},
    {"file": "Accumulation-Conservative.csv", "name": "Conservative"},
    {"file": "Accumulation-Defensive.csv", "name": "Defensive"},
    {"file": "Accumulation-Australian-Shares.csv", "name": "Australian Shares"},
    {"file": "Accumulation-International-Shares.csv", "name": "International Shares"},
    {"file": "Accumulation-Property.csv", "name": "Property"},
    {"file": "Accumulation-Bonds.csv", "name": "Bonds"},
    {"file": "Accumulation-Term-Deposit.csv", "name": "Term Deposit"},
    {"file": "Accumulation-Cash.csv", "name": "Cash"}
]

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="FutureSaver Analysis")
st.title("FutureSaver - News, Sentiment & Impact Estimator")
st.caption(f"Report Generated: {datetime.now().strftime('%A, %d %B %Y, %I:%M %p AEST')}")

st.sidebar.header("Data Upload")
# Define uploaded_unit_price_file at the top of the script execution related to UI
uploaded_unit_price_file = st.sidebar.file_uploader("Upload Latest Unit Price History CSV", type=["csv"])

# --- Google Sheets Helper Functions ---
@st.cache_resource(ttl=600) 
def init_gspread_client():
    try:
        # Try to get secrets from Streamlit Cloud's st.secrets
        if hasattr(st, 'secrets') and "google_sheets_credentials" in st.secrets:
            creds_input = st.secrets["google_sheets_credentials"]
            # Streamlit secrets can parse TOML strings directly into dicts if simple,
            # or pass the string representation of the JSON.
            if isinstance(creds_input, str): # If it's a string, assume it's JSON string
                creds_dict = json.loads(creds_input)
            elif isinstance(creds_input, dict): # If Streamlit already parsed it as dict
                creds_dict = creds_input
            else:
                st.error("Google Sheets credentials in Streamlit secrets are not in a recognized format (string or dict).")
                return None
            creds = Credentials.from_service_account_info(creds_dict)
        else: # Fallback for local development: try to load from .streamlit/secrets.toml
            # This assumes you have a .streamlit/secrets.toml file in your project root locally
            # with the google_sheets_credentials formatted as a TOML string.
            # For local, gspread can also try to find default credentials if you've run `gcloud auth application-default login`
            # but explicit service account is better for consistency.
            # For simplicity, we'll rely on st.secrets and expect it to work or the user to have local TOML.
            # If st.secrets doesn't exist locally, this will error out, prompting local setup.
            st.warning("Attempting to use Google Sheets without Streamlit Cloud secrets. "
                       "Ensure local `secrets.toml` is configured if not deployed, or app may fail.")
            # This path is illustrative of how gspread might find default creds;
            # robust local secret handling might require explicitly reading a local JSON file.
            # For now, let's just let gspread try its default auth flow if st.secrets fails locally.
            # OR, be more explicit for local:
            local_secrets_path = os.path.join(".streamlit", "secrets.toml")
            if os.path.exists(local_secrets_path):
                import toml
                secrets_local = toml.load(local_secrets_path)
                if "google_sheets_credentials" in secrets_local:
                    creds_input = secrets_local["google_sheets_credentials"]
                    if isinstance(creds_input, str):
                        creds_dict = json.loads(creds_input)
                    elif isinstance(creds_input, dict):
                         creds_dict = creds_input
                    else:
                        st.error("Local Google Sheets credentials are not in a recognized format.")
                        return None
                    creds = Credentials.from_service_account_info(creds_dict)
                else:
                    st.error("`google_sheets_credentials` not found in local `.streamlit/secrets.toml`.")
                    return None
            else:
                # If running locally without secrets.toml, gspread.service_account() can be used
                # if the JSON key file is locally available, but that's less secure to hardcode path.
                # Best to instruct user to create .streamlit/secrets.toml for local use.
                st.error("Local `.streamlit/secrets.toml` not found for Google Sheets credentials.")
                return None


        scoped_creds = creds.with_scopes([
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file" 
        ])
        client = gspread.authorize(scoped_creds)
        print("Successfully initialized gspread client.")
        return client
    except Exception as e:
        st.error(f"Google Sheets Error: Could not initialize gspread client: {e}")
        print(f"Error initializing gspread client: {e}")
        return None

# ... (load_estimates_from_gsheet and save_estimate_to_gsheet remain the same as main_app_py_v10) ...
def load_estimates_from_gsheet(client, sheet_name, worksheet_name):
    expected_cols = ['RunDateTime', 'EstimationForDate', 'FundName', 
                     'EstimatedUnitPrice', 'BaselinePriceUsed', 'TotalEstPercentChange']
    if client is None:
        return pd.DataFrame(columns=expected_cols)
    try:
        sheet = client.open(sheet_name).worksheet(worksheet_name)
        df = get_as_dataframe(sheet, evaluate_formulas=False, include_index=False, header=0, na_filter=False) # na_filter=False to get empty strings as is
        
        if not df.empty:
            # Convert to appropriate types, handling empty strings explicitly before conversion
            if 'RunDateTime' in df.columns:
                df['RunDateTime'] = pd.to_datetime(df['RunDateTime'].replace('', None), errors='coerce')
            if 'EstimationForDate' in df.columns:
                df['EstimationForDate'] = pd.to_datetime(df['EstimationForDate'].replace('', None), errors='coerce')
            
            for col in ['EstimatedUnitPrice', 'BaselinePriceUsed', 'TotalEstPercentChange']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].replace('', None), errors='coerce')
            
            df.dropna(subset=['RunDateTime', 'EstimationForDate', 'FundName'], how='all', inplace=True) 
        else: 
            df = pd.DataFrame(columns=expected_cols)
        
        # Ensure all expected columns exist, even if empty, after loading
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype='object' if 'Date' in col else ('float64' if 'Price' in col or 'Change' in col else 'object'))


        print(f"Successfully loaded {len(df)} estimates from Google Sheet: {sheet_name}/{worksheet_name}")
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Google Sheet '{sheet_name}' not found. Please ensure it exists and is shared with the service account.")
        return pd.DataFrame(columns=expected_cols)
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Worksheet '{worksheet_name}' not found in Google Sheet '{sheet_name}'.")
        return pd.DataFrame(columns=expected_cols)
    except Exception as e:
        st.error(f"Google Sheets Error: Could not load estimates: {e}")
        print(f"Error loading estimates from GSheet: {e}")
        return pd.DataFrame(columns=expected_cols)

def save_estimate_to_gsheet(client, sheet_name, worksheet_name, current_log_df, new_estimate_data):
    if client is None:
        st.warning("Google Sheets client not initialized. Cannot save estimate to cloud.")
        return current_log_df 

    try:
        sheet = client.open(sheet_name).worksheet(worksheet_name)
        
        # Prepare the new row for DataFrame creation
        new_row_dict_for_df = {k: [v] for k,v in new_estimate_data.items()}
        new_row_df = pd.DataFrame(new_row_dict_for_df)
        
        # Convert datetimes in new_row_df to ISO format strings for gspread
        if 'RunDateTime' in new_row_df.columns:
            new_row_df['RunDateTime'] = pd.to_datetime(new_row_df['RunDateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'EstimationForDate' in new_row_df.columns:
            new_row_df['EstimationForDate'] = pd.to_datetime(new_row_df['EstimationForDate']).dt.strftime('%Y-%m-%d')

        # Ensure current_log_df has dtypes consistent for concat, or reload fresh before concat
        # For simplicity, we will append the new row directly to the sheet if possible,
        # or reload, concat, and overwrite. set_with_dataframe is safer.

        # Ensure columns match before concat, especially if current_log_df was empty with specific dtypes
        # It's safer to convert the current_log_df datetimes to strings too before concat & save
        df_to_save = current_log_df.copy()
        if 'RunDateTime' in df_to_save.columns:
            df_to_save['RunDateTime'] = pd.to_datetime(df_to_save['RunDateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'EstimationForDate' in df_to_save.columns:
            df_to_save['EstimationForDate'] = pd.to_datetime(df_to_save['EstimationForDate']).dt.strftime('%Y-%m-%d')
        
        # Ensure new_row_df has all columns that df_to_save has, filling with None or appropriate default if not
        for col in df_to_save.columns:
            if col not in new_row_df.columns:
                new_row_df[col] = None # Or pd.NA

        # Reorder columns of new_row_df to match df_to_save for clean concat
        if not df_to_save.empty: # Only if there's existing data to match columns with
             new_row_df = new_row_df[df_to_save.columns]


        updated_df_for_gsheet = pd.concat([df_to_save, new_row_df], ignore_index=True)
        
        # Ensure all data is string for gspread_dataframe (it handles nan/None to empty string)
        set_with_dataframe(sheet, updated_df_for_gsheet.astype(str), include_index=False, resize=True) 
        
        print(f"Successfully saved estimate for {new_estimate_data['FundName']} to GSheet: {sheet_name}/{worksheet_name}")
        # Return the DataFrame with proper dtypes after reloading
        return load_estimates_from_gsheet(client, sheet_name, worksheet_name) 

    except Exception as e:
        st.error(f"Google Sheets Error: Could not save estimate: {e}")
        print(f"Error saving estimate to GSheet: {e}")
        return current_log_df

# --- Initialize Gspread Client & Load Estimates Log ---
# This is now called *before* load_all_fund_data, which is fine.
gs_client = init_gspread_client()
estimates_log_df_global = load_estimates_from_gsheet(gs_client, GOOGLE_SHEET_NAME, WORKSHEET_NAME)


# --- Global Data Loading (cached) ---
# The uploaded_unit_price_file variable is defined by st.sidebar.file_uploader at the top execution of the script
@st.cache_data(ttl=3600) 
def load_all_fund_data(uploaded_file_obj_param): # Renamed param to avoid conflict with global
    price_history_df = pd.DataFrame() 
    if uploaded_file_obj_param is not None:
        price_history_df = load_unit_price_history(uploaded_file_obj_param)
        if 'Date' in price_history_df.columns:
            price_history_df['Date'] = pd.to_datetime(price_history_df['Date'], errors='coerce')
            price_history_df.dropna(subset=['Date'], inplace=True)
        # st.sidebar.success(f"Loaded unit prices from: {uploaded_file_obj_param.name}") # Done by uploader
    else:
        # st.sidebar.warning("Please upload a Unit Price History CSV to begin analysis.") # Done by uploader
        default_unit_price_file = os.path.join(DATA_DIR, "FutureSaver_UnitPriceHistory-22-05-2025.csv") 
        if os.path.exists(default_unit_price_file):
            price_history_df = load_unit_price_history(default_unit_price_file)
            if 'Date' in price_history_df.columns:
                 price_history_df['Date'] = pd.to_datetime(price_history_df['Date'], errors='coerce')
                 price_history_df.dropna(subset=['Date'], inplace=True)
            # st.sidebar.info(f"Using default unit price file from local 'data' folder: {default_unit_price_file}")
        else:
             price_history_df = pd.DataFrame(columns=['Date', 'FundName', 'UnitPrice'])

    all_fund_holdings_data = {}
    combined_keywords = set()
    for fund_spec in HOLDINGS_FILES_INFO:
        # ... (rest of holdings loading and keyword generation as in main_app_py_v10) ...
        holdings_file_path = os.path.join(DATA_DIR, fund_spec["file"])
        if not os.path.exists(holdings_file_path):
            print(f"Holdings file not found: {holdings_file_path} for fund {fund_spec['name']}")
            all_fund_holdings_data[fund_spec["name"]] = pd.DataFrame() 
            continue
        holdings_df = load_fund_holdings(holdings_file_path) 
        all_fund_holdings_data[fund_spec["name"]] = holdings_df
        if not holdings_df.empty:
            if "CanonicalHoldingName" in holdings_df.columns:
                for name in holdings_df["CanonicalHoldingName"].dropna().unique():
                    cleaned_name = str(name).lower().strip()
                    if cleaned_name and cleaned_name not in ['nan', '-', 'unspecified holding', '']: 
                        combined_keywords.add(cleaned_name)
            if "Ticker" in holdings_df.columns:
                for ticker in holdings_df["Ticker"].dropna().unique():
                    cleaned_ticker = str(ticker).lower().strip()
                    if cleaned_ticker and cleaned_ticker not in ['nan', '-', '']: 
                        combined_keywords.add(cleaned_ticker)
    
    general_market_keywords = { 
        "interest rate", "rba cash rate", "monetary policy", "inflation", "cpi",
        "australian property market", "housing prices", "construction industry",
        "asx 200", "s&p 500", "global economy", "aud usd"
    }
    combined_keywords.update(general_market_keywords)
    combined_keywords = {kw for kw in combined_keywords if kw and kw != 'nan' and kw != '-'} 
    return price_history_df, all_fund_holdings_data, list(combined_keywords)

# --- Call load_all_fund_data with the uploader variable ---
price_data, holdings_data_all_funds, all_keywords = load_all_fund_data(uploaded_unit_price_file)


# ... (Rest of the main_app.py script from v10 continues here, using price_data, 
#      holdings_data_all_funds, all_keywords, and estimates_log_df_global for plotting,
#      and calling save_estimate_to_gsheet to update estimates_log_df_global) ...

if price_data.empty and not uploaded_unit_price_file: # Check if still no data
    st.error("Please upload a Unit Price History CSV file to start the analysis.")
    # Consider whether to st.stop() here or allow app to run in limited state
    # For now, let's allow it to proceed to see if holdings load for keyword generation
    # but many features will be disabled.

news_items_analyzed = []
if all_keywords:
    news_items_analyzed = fetch_process_news(all_keywords)

st.sidebar.header("Fund Selection")
available_fund_names = sorted([f_info["name"] for f_info in HOLDINGS_FILES_INFO if f_info["name"] in holdings_data_all_funds and not holdings_data_all_funds[f_info["name"]].empty])

if not available_fund_names:
    st.error("No fund data loaded successfully for selection. Check console and ensure holdings files are in 'data' folder.")
    st.stop() # Stop if no funds can be selected

selected_fund_for_detail = st.sidebar.selectbox("Choose a fund for detailed analysis:", available_fund_names)
st.header(f"Detailed Analysis for: {selected_fund_for_detail}")

# --- Display Latest Unit Price & ACTUAL Day-over-Day Change ---
last_price = 0.0
previous_price = 0.0
last_price_date = None 
last_price_date_str = "N/A"

if not price_data.empty:
    fund_price_history_selected = price_data[price_data['FundName'] == selected_fund_for_detail].sort_values(by='Date', ascending=False)
    if not fund_price_history_selected.empty:
        latest_entry = fund_price_history_selected.iloc[0]
        last_price = latest_entry['UnitPrice']
        last_price_date = latest_entry['Date'] 
        last_price_date_str = last_price_date.strftime('%d/%m/%Y')
        st.subheader(f"Latest Actual Unit Price ({last_price_date_str}): {last_price:.{UNIT_PRICE_DISPLAY_PRECISION}f}")

        if len(fund_price_history_selected) > 1:
            previous_entry = fund_price_history_selected.iloc[1]
            if previous_entry['Date'] < last_price_date: # Ensure it's actually a previous date
                previous_price = previous_entry['UnitPrice']
                previous_price_date_str = previous_entry['Date'].strftime('%d/%m/%Y')
                actual_change_val = last_price - previous_price
                actual_change_percent = (actual_change_val / previous_price) * 100 if previous_price != 0 else 0
                actual_change_str = f"{actual_change_val:+.{UNIT_PRICE_DISPLAY_PRECISION}f}"
                actual_change_percent_str = f"{actual_change_percent:+.4f}%"
                st.metric(label=f"Actual Change from {previous_price_date_str}", value=actual_change_str, delta=f"{actual_change_percent_str}")
            else:
                st.info("Previous day's price not directly before latest; using latest as baseline for estimation.")
                previous_price = last_price # Use last_price as baseline if no valid previous
        else:
            st.info("Not enough historical data (need at least 2 records) to calculate actual day-over-day change.")
            previous_price = last_price 
    else: 
        st.warning(f"No price history found for {selected_fund_for_detail} in the uploaded file.")
else: 
    st.warning("Unit price history data is not available or not loaded.")

current_holdings = holdings_data_all_funds.get(selected_fund_for_detail)
baseline_price_for_estimation = previous_price if previous_price > 0 else last_price 

if current_holdings is not None and not current_holdings.empty and news_items_analyzed and baseline_price_for_estimation > 0 and last_price_date is not None:
    st.subheader(f"Impact Estimation on Unit Price (Estimating for {last_price_date.strftime('%d/%m/%Y')} based on news up to today):")
    with st.spinner("Calculating estimated impact..."):
        fund_specific_keywords_lc_iem = set()
        if "CanonicalHoldingName" in current_holdings.columns:
            for name_val in current_holdings["CanonicalHoldingName"].dropna().unique():
                cleaned_name = str(name_val).lower().strip()
                if cleaned_name and cleaned_name not in ['nan', '-', 'unspecified holding', '']: fund_specific_keywords_lc_iem.add(cleaned_name)
        if "Ticker" in current_holdings.columns:
            for ticker_val in current_holdings["Ticker"].dropna().unique():
                cleaned_ticker = str(ticker_val).lower().strip()
                if cleaned_ticker and cleaned_ticker not in ['nan', '-', '']: fund_specific_keywords_lc_iem.add(cleaned_ticker)
        
        general_keywords_for_iem = set()
        if "AssetClass" in current_holdings.columns:
            fund_asset_classes_lc_iem = {str(ac).lower() for ac in current_holdings["AssetClass"].unique()}
            if any("property" in ac for ac in fund_asset_classes_lc_iem): general_keywords_for_iem.update(["property market", "housing price", "construction industry"])
            if any(ac for ac in fund_asset_classes_lc_iem if "bond" in ac or "fixed income" in ac or "fixed interest" in ac): general_keywords_for_iem.update(["interest rate", "rba", "monetary policy", "inflation", "cpi"])
        keywords_for_iem_filter = fund_specific_keywords_lc_iem.union(general_keywords_for_iem)
        
        news_for_iem_calculation = []
        for news_item_full in news_items_analyzed:
            original_match_keyword = str(news_item_full.get("matched_keyword", "")).lower().strip()
            if original_match_keyword in keywords_for_iem_filter:
                news_for_iem_calculation.append(news_item_full)

        if news_for_iem_calculation:
            iem_result = calculate_fund_impact(current_holdings, news_for_iem_calculation, baseline_price_for_estimation)
            estimated_unit_price = iem_result.get('estimated_new_unit_price', baseline_price_for_estimation)
            total_est_percent_change = iem_result.get('total_estimated_fund_percentage_change',0.0)

            st.metric("Total Est. % Change from News", f"{total_est_percent_change:.4f}%")
            st.metric(f"Estimated Unit Price (for {last_price_date.strftime('%d/%m/%Y')})", f"{estimated_unit_price:.{UNIT_PRICE_DISPLAY_PRECISION}f}")

            if last_price_date and gs_client: # Ensure gs_client is initialized 
                new_estimate_data = {
                    'RunDateTime': datetime.now(), 
                    'EstimationForDate': last_price_date, 
                    'FundName': selected_fund_for_detail,
                    'EstimatedUnitPrice': estimated_unit_price,
                    'BaselinePriceUsed': baseline_price_for_estimation,
                    'TotalEstPercentChange': total_est_percent_change
                }
                estimates_log_df_global = save_estimate_to_gsheet(gs_client, GOOGLE_SHEET_NAME, WORKSHEET_NAME, estimates_log_df_global, new_estimate_data)
                
            with st.expander("Show Detailed Impact Calculations"):
                impact_details_list = iem_result.get('impact_details', [])
                if impact_details_list:
                    for detail in impact_details_list:
                        st.markdown(f"""
                        **Holding/Topic:** {detail.get('holding_name','N/A')}
                        * News: '{detail.get('news_title','N/A')}' (Sentiment: {detail.get('sentiment_label','N/A')}, Score: {detail.get('sentiment_score',0.0):.2f})
                        * Est. Impact on Asset/Topic: {detail.get('est_impact_on_holding_price_percent',0.0):.2f}%
                        * Contribution to Fund Change: {detail.get('contribution_to_fund_percent_change',0.0):.4f}%
                        ---
                        """)
                else: st.write("No news items directly contributed to a calculated impact for this fund.")
        else:
            st.info("No news items (after filtering for this specific fund) to calculate an impact.")

elif baseline_price_for_estimation == 0 and current_holdings is not None and not current_holdings.empty: 
    st.warning("Cannot calculate impact: Baseline unit price for estimation is zero or unavailable.")

st.markdown("---") 
# ... (Top 5 Holdings display as in v8) ...
if current_holdings is not None and not current_holdings.empty:
    st.subheader("Top 5 Holdings:")
    if "Weighting" in current_holdings.columns:
        st.dataframe(current_holdings.nlargest(5, 'Weighting')[["CanonicalHoldingName", "Ticker", "AssetClass", "Weighting", "Currency"]])
    else:
        st.warning("Top holdings display issue: 'Weighting' column missing.")
        cols_to_show = [col for col in ["CanonicalHoldingName", "Ticker", "AssetClass", "Currency"] if col in current_holdings.columns]
        if cols_to_show: st.dataframe(current_holdings.head()[cols_to_show])
else:
    st.warning(f"No holdings data loaded for {selected_fund_for_detail}.")

# --- Historical Unit Price Chart (Actual vs. Estimated from GSheet) ---
st.markdown("---")
st.header("Historical Unit Price Chart (Actual vs. Estimated)")

# Use estimates_log_df_global which is loaded from GSheet at app start and updated in-session
if not price_data.empty and not estimates_log_df_global.empty:
    all_fund_names_for_plot = sorted(list(set(price_data['FundName'].unique()) | set(estimates_log_df_global['FundName'].unique())))
    
    default_selection_for_plot = [selected_fund_for_detail] if selected_fund_for_detail in all_fund_names_for_plot else ([all_fund_names_for_plot[0]] if all_fund_names_for_plot else [])

    selected_funds_for_plot = st.multiselect(
        "Select fund(s) to plot:",
        options=all_fund_names_for_plot,
        default=default_selection_for_plot
    )

    if selected_funds_for_plot:
        # Prepare Actual Prices Data
        actual_plot_data = price_data[price_data['FundName'].isin(selected_funds_for_plot)].copy()
        actual_plot_pivot = pd.DataFrame()
        if not actual_plot_data.empty:
            actual_plot_data['Date'] = pd.to_datetime(actual_plot_data['Date'])
            actual_plot_pivot = actual_plot_data.pivot_table(
                index='Date', columns='FundName', values='UnitPrice'
            ).sort_index()
            actual_plot_pivot.columns = [f"{col}_Actual" for col in actual_plot_pivot.columns]
            
        # Prepare Estimated Prices Data
        estimated_plot_data_source = estimates_log_df_global[estimates_log_df_global['FundName'].isin(selected_funds_for_plot)].copy()
        estimated_plot_pivot = pd.DataFrame()
        if not estimated_plot_data_source.empty:
            if 'EstimationForDate' in estimated_plot_data_source.columns:
                 estimated_plot_data_source['EstimationForDate'] = pd.to_datetime(estimated_plot_data_source['EstimationForDate'])
            
            # Taking the latest estimate if multiple exist for the same day for the same fund
            estimated_plot_data_agg = estimated_plot_data_source.sort_values('RunDateTime').groupby(
                ['EstimationForDate', 'FundName'], as_index=False
            ).last() # Takes the last entry based on original sort (RunDateTime)
            
            estimated_plot_pivot = estimated_plot_data_agg.pivot_table(
                index='EstimationForDate', columns='FundName', values='EstimatedUnitPrice'
            ).sort_index()
            estimated_plot_pivot.columns = [f"{col}_Estimated" for col in estimated_plot_pivot.columns]
            estimated_plot_pivot.index.name = 'Date' 

        # Merge actual and estimated data
        if not actual_plot_pivot.empty and not estimated_plot_pivot.empty:
            combined_plot_df = pd.merge(actual_plot_pivot, estimated_plot_pivot, on='Date', how='outer').sort_index()
        elif not actual_plot_pivot.empty:
            combined_plot_df = actual_plot_pivot
        elif not estimated_plot_pivot.empty:
            combined_plot_df = estimated_plot_pivot
        else:
            combined_plot_df = pd.DataFrame() # Should not happen if selection made and data exists
        
        if not combined_plot_df.empty:
            cols_to_plot = []
            for fund_name_plot in selected_funds_for_plot:
                actual_col_name = f"{fund_name_plot}_Actual"
                estimated_col_name = f"{fund_name_plot}_Estimated"
                if actual_col_name in combined_plot_df.columns: cols_to_plot.append(actual_col_name)
                if estimated_col_name in combined_plot_df.columns: cols_to_plot.append(estimated_col_name)
            
            if cols_to_plot:
                st.line_chart(combined_plot_df[cols_to_plot])
            else: st.write("No data columns available for plotting for the selected fund(s).")
        else: st.write("No data for selected fund(s) to plot (actual or estimated).")
    else: st.info("Select one or more funds to display their historical prices.")
elif price_data.empty:
    st.warning("Unit price data not loaded. Cannot display historical chart.")
elif estimates_log_df_global.empty:
    st.info("No estimates have been logged yet to display on the historical chart.")


# --- Display Relevant News (as in v8) ---
# ... (News display section remains the same as main_app_py_v8) ...
if current_holdings is not None and not current_holdings.empty:
    fund_specific_keywords_lc_display = set()
    if "CanonicalHoldingName" in current_holdings.columns:
        for name_val in current_holdings["CanonicalHoldingName"].dropna().unique():
            cleaned_name = str(name_val).lower().strip()
            if cleaned_name and cleaned_name not in ['nan', '-', 'unspecified holding', '']: fund_specific_keywords_lc_display.add(cleaned_name)
    if "Ticker" in current_holdings.columns:
        for ticker_val in current_holdings["Ticker"].dropna().unique():
            cleaned_ticker = str(ticker_val).lower().strip()
            if cleaned_ticker and cleaned_ticker not in ['nan', '-', '']: fund_specific_keywords_lc_display.add(cleaned_ticker)
    
    general_keywords_for_display = set()
    if "AssetClass" in current_holdings.columns:
        fund_asset_classes_lc_display = {str(ac).lower() for ac in current_holdings["AssetClass"].unique()}
        if any("property" in ac for ac in fund_asset_classes_lc_display): general_keywords_for_display.update(["property market", "housing price", "construction industry"])
        if any(ac for ac in fund_asset_classes_lc_display if "bond" in ac or "fixed income" in ac or "fixed interest" in ac): general_keywords_for_display.update(["interest rate", "rba", "monetary policy", "inflation", "cpi"])
    display_keywords_for_fund_news = fund_specific_keywords_lc_display.union(general_keywords_for_display)
    
    fund_relevant_news_display_list = []
    processed_news_links_for_display_set = set()
    for news_item_full_display in news_items_analyzed:
        text_content_display = (str(news_item_full_display.get("title","")) + " " + str(news_item_full_display.get("summary",""))).lower()
        news_matched_display_keyword = None
        for kw_disp in display_keywords_for_fund_news: 
            if kw_disp in text_content_display:
                if kw_disp in general_keywords_for_display and kw_disp not in fund_specific_keywords_lc_display:
                    news_matched_display_keyword = f"General: {kw_disp}"
                else:
                    news_matched_display_keyword = kw_disp 
                break 
        
        if news_matched_display_keyword and news_item_full_display.get('link') not in processed_news_links_for_display_set:
            news_copy_display = news_item_full_display.copy()
            news_copy_display["display_matched_keyword_context"] = news_matched_display_keyword 
            fund_relevant_news_display_list.append(news_copy_display)
            if news_item_full_display.get('link'): 
                processed_news_links_for_display_set.add(news_item_full_display['link'])
            
    st.subheader(f"Recent Relevant News ({len(fund_relevant_news_display_list)} items found for this fund):")
    if fund_relevant_news_display_list:
        for item_disp in fund_relevant_news_display_list[:20]: 
            exp_title_disp = f"{item_disp.get('published_str','N/A')} - **{item_disp.get('sentiment_strength','N/A')} ({item_disp.get('sentiment_compound',0.0):.2f})**: {item_disp.get('title','No Title')}"
            with st.expander(exp_title_disp):
                st.markdown(f"**Source:** {item_disp.get('source_feed_name','N/A')}")
                st.markdown(f"**Matched Context for Display:** {item_disp.get('display_matched_keyword_context', item_disp.get('matched_keyword', 'N/A'))}")
                st.markdown(str(item_disp.get("summary","")))
                if item_disp.get('link'): st.markdown(f"[Read full article]({item_disp['link']})", unsafe_allow_html=True)
    else: st.info("No specific or generally relevant news found for this fund in the last 3 days from configured feeds for display.")

st.sidebar.markdown("---")
st.sidebar.info("Prototype V1.0. For informational and educational purposes only. Not financial advice.")

