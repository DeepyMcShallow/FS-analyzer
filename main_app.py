# main_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import io 
import gspread # For Google Sheets
from gspread_dataframe import get_as_dataframe, set_with_dataframe # For pandas integration
from google.oauth2.service_account import Credentials # For authentication

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
# ESTIMATES_LOG_FILE = os.path.join(DATA_DIR, "estimated_unit_prices_log.csv") # No longer primary
GOOGLE_SHEET_NAME = "FutureSaverEstimatesLog"  # The EXACT name of your Google Sheet
WORKSHEET_NAME = "Sheet1" # Usually the default first sheet, rename if needed

HOLDINGS_FILES_INFO = [ 
    {"file": "Accumulation-High-Growth.csv", "name": "High Growth"},
    # ... (rest of your HOLDINGS_FILES_INFO remains the same) ...
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

# --- Google Sheets Helper Functions ---
@st.cache_resource(ttl=600) # Cache client for 10 minutes
def init_gspread_client():
    """Initializes and returns the gspread client using Streamlit Secrets."""
    try:
        # For Streamlit Cloud, credentials come from st.secrets
        # Ensure your secret is named "google_sheets_credentials" or adjust here
        creds_json = st.secrets["google_sheets_credentials"]
        
        # If creds_json is already a dict (TOML parsed by Streamlit)
        if isinstance(creds_json, dict):
            creds = Credentials.from_service_account_info(creds_json)
        else: # If it's a string (e.g. direct JSON string in local .streamlit/secrets.toml)
            import json
            creds_dict = json.loads(creds_json)
            creds = Credentials.from_service_account_info(creds_dict)
        
        scoped_creds = creds.with_scopes([
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file" # Often needed
        ])
        client = gspread.authorize(scoped_creds)
        print("Successfully initialized gspread client.")
        return client
    except Exception as e:
        st.error(f"Google Sheets Error: Could not initialize gspread client: {e}")
        print(f"Error initializing gspread client: {e}")
        # You might want to check st.secrets contents if this fails
        # print(f"DEBUG: Type of st.secrets['google_sheets_credentials']: {type(st.secrets.get('google_sheets_credentials'))}")
        return None

def load_estimates_from_gsheet(client, sheet_name, worksheet_name):
    """Loads estimates from the specified Google Sheet into a DataFrame."""
    expected_cols = ['RunDateTime', 'EstimationForDate', 'FundName', 
                     'EstimatedUnitPrice', 'BaselinePriceUsed', 'TotalEstPercentChange']
    if client is None:
        return pd.DataFrame(columns=expected_cols)
    try:
        sheet = client.open(sheet_name).worksheet(worksheet_name)
        # Use include_index=False, evaluate_formulas=False for robustness
        df = get_as_dataframe(sheet, evaluate_formulas=False, include_index=False, header=0)
        
        # Ensure correct dtypes, especially for dates
        if not df.empty:
            if 'RunDateTime' in df.columns:
                df['RunDateTime'] = pd.to_datetime(df['RunDateTime'], errors='coerce')
            if 'EstimationForDate' in df.columns:
                df['EstimationForDate'] = pd.to_datetime(df['EstimationForDate'], errors='coerce')
            # Ensure numeric columns are numeric
            for col in ['EstimatedUnitPrice', 'BaselinePriceUsed', 'TotalEstPercentChange']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['RunDateTime', 'EstimationForDate', 'FundName'], inplace=True) # Essential columns
        else: # Sheet might be empty but headers exist, or completely empty
            # If df is empty but sheet.row_count > 0, it implies headers might be there.
            # If sheet is truly empty, get_as_dataframe might return empty df with no columns.
            # So, ensure columns if empty.
            df = pd.DataFrame(columns=expected_cols)


        print(f"Successfully loaded {len(df)} estimates from Google Sheet: {sheet_name}/{worksheet_name}")
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Google Sheet '{sheet_name}' not found. Please ensure it exists and is shared with the service account.")
        print(f"Error: Google Sheet '{sheet_name}' not found.")
        return pd.DataFrame(columns=expected_cols)
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Worksheet '{worksheet_name}' not found in Google Sheet '{sheet_name}'.")
        print(f"Error: Worksheet '{worksheet_name}' not found in '{sheet_name}'.")
        return pd.DataFrame(columns=expected_cols)
    except Exception as e:
        st.error(f"Google Sheets Error: Could not load estimates: {e}")
        print(f"Error loading estimates from GSheet: {e}")
        return pd.DataFrame(columns=expected_cols)

def save_estimate_to_gsheet(client, sheet_name, worksheet_name, current_log_df, new_estimate_data):
    """Appends a new estimate to the Google Sheet and returns the updated DataFrame."""
    if client is None:
        st.warning("Google Sheets client not initialized. Cannot save estimate.")
        return current_log_df # Return the in-memory version

    try:
        sheet = client.open(sheet_name).worksheet(worksheet_name)
        new_row_df = pd.DataFrame([new_estimate_data])
        
        # Convert datetimes to ISO format strings for gspread, or gspread-dataframe might handle it
        new_row_df['RunDateTime'] = pd.to_datetime(new_row_df['RunDateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        new_row_df['EstimationForDate'] = pd.to_datetime(new_row_df['EstimationForDate']).dt.strftime('%Y-%m-%d')

        # Basic duplicate check (optional, can be more sophisticated)
        # This check is simpler if we assume datetimes in current_log_df are already pd.to_datetime
        # For simplicity, let's append and let Google Sheets handle multiple entries if any,
        # or rely on the plotting logic to take the mean/last for a given day.

        # Append the new row using gspread's append_row or set_with_dataframe
        # Using set_with_dataframe to update the whole sheet is safer if headers might be missing
        # Or, if sheet is guaranteed to have headers:
        # sheet.append_row(new_row_df.iloc[0].values.tolist())

        # More robust: get current data, append, then set whole sheet
        # This also ensures headers are written if the sheet was completely empty
        updated_df_for_gsheet = pd.concat([current_log_df, new_row_df], ignore_index=True)
        # Convert datetimes to strings before writing to avoid gspread issues with tz-aware/naive
        if 'RunDateTime' in updated_df_for_gsheet.columns:
            updated_df_for_gsheet['RunDateTime'] = pd.to_datetime(updated_df_for_gsheet['RunDateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'EstimationForDate' in updated_df_for_gsheet.columns:
            updated_df_for_gsheet['EstimationForDate'] = pd.to_datetime(updated_df_for_gsheet['EstimationForDate']).dt.strftime('%Y-%m-%d')

        set_with_dataframe(sheet, updated_df_for_gsheet, include_index=False, resize=True)
        
        print(f"Successfully saved estimate for {new_estimate_data['FundName']} to GSheet: {sheet_name}/{worksheet_name}")
        # Return the DataFrame with proper dtypes for in-memory use
        return load_estimates_from_gsheet(client, sheet_name, worksheet_name) 

    except Exception as e:
        st.error(f"Google Sheets Error: Could not save estimate: {e}")
        print(f"Error saving estimate to GSheet: {e}")
        return current_log_df # Return the original if save failed


# --- Initialize Gspread Client & Load Estimates Log from GSheet ---
gs_client = init_gspread_client()
estimates_log_df = load_estimates_from_gsheet(gs_client, GOOGLE_SHEET_NAME, WORKSHEET_NAME)


# --- Global Data Loading (cached) ---
@st.cache_data(ttl=3600) 
def load_all_fund_data(uploaded_file_obj):
    # ... (load_all_fund_data function remains mostly the same as main_app_py_v8) ...
    # ... but ensure it doesn't rely on local estimates_log_df ...
    price_history_df = pd.DataFrame() 
    if uploaded_file_obj is not None:
        price_history_df = load_unit_price_history(uploaded_file_obj)
        if 'Date' in price_history_df.columns:
            price_history_df['Date'] = pd.to_datetime(price_history_df['Date'], errors='coerce')
            price_history_df.dropna(subset=['Date'], inplace=True)
        st.sidebar.success(f"Loaded unit prices from: {uploaded_file_obj.name}")
    else:
        st.sidebar.warning("Please upload a Unit Price History CSV to begin analysis.")
        default_unit_price_file = os.path.join(DATA_DIR, "FutureSaver_UnitPriceHistory-22-05-2025.csv") 
        if os.path.exists(default_unit_price_file):
            price_history_df = load_unit_price_history(default_unit_price_file)
            if 'Date' in price_history_df.columns:
                 price_history_df['Date'] = pd.to_datetime(price_history_df['Date'], errors='coerce')
                 price_history_df.dropna(subset=['Date'], inplace=True)
            st.sidebar.info(f"Using default unit price file from local 'data' folder: {default_unit_price_file}")
        else:
             price_history_df = pd.DataFrame(columns=['Date', 'FundName', 'UnitPrice'])

    all_fund_holdings_data = {}
    combined_keywords = set()
    for fund_spec in HOLDINGS_FILES_INFO:
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

@st.cache_data(ttl=900) 
def fetch_process_news(keywords_list):
    # ... (fetch_process_news function remains the same as main_app_py_v8) ...
    raw_news = aggregate_news(RSS_FEEDS_CONFIG, keywords_list, days_history=3) 
    analyzed_news = []
    for item in raw_news:
        title_str = str(item.get("title", ""))
        summary_str = str(item.get("summary", ""))
        sentiment = get_sentiment(title_str + " " + summary_str)
        item["sentiment_label"] = sentiment["label"]
        item["sentiment_compound"] = sentiment["compound"]
        item["sentiment_strength"] = classify_sentiment_strength(sentiment["compound"])
        analyzed_news.append(item)
    return analyzed_news

# --- Main App Logic ---
price_data, holdings_data_all_funds, all_keywords = load_all_fund_data(uploaded_unit_price_file)

if price_data.empty and not uploaded_unit_price_file:
    st.error("Please upload a Unit Price History CSV file to start the analysis.")
    st.stop()
# ... (rest of main_app.py logic from v8 continues here, with one key change noted below) ...

# --- Key Change: Inside the IEM Display Block, when saving the estimate ---
# This block is where iem_result is available
# if news_for_iem_calculation:
#    iem_result = calculate_fund_impact(...)
#    ... (display metrics) ...
#    if last_price_date: # Ensure we have a valid date for the estimation
#        new_estimate_data = {
#            'RunDateTime': datetime.now(), # This is correct
#            'EstimationForDate': last_price_date, # This is correct
#            'FundName': selected_fund_for_detail,
#            'EstimatedUnitPrice': estimated_unit_price, # from iem_result
#            'BaselinePriceUsed': baseline_price_for_estimation,
#            'TotalEstPercentChange': total_est_percent_change # from iem_result
#        }
#        # **MODIFICATION HERE:**
#        # Call the new Google Sheets save function
#        estimates_log_df = save_estimate_to_gsheet(gs_client, GOOGLE_SHEET_NAME, WORKSHEET_NAME, estimates_log_df, new_estimate_data)
#        # The global estimates_log_df is updated for the current session for plotting

# --- Key Change: In the Historical Unit Price Chart section ---
# Instead of: current_estimates_log_df = load_or_initialize_estimates_log()
# Use the global estimates_log_df which is now sourced from GSheet at app start
# and updated in-session after each new estimate is saved to GSheet.
#
# if not price_data.empty:
#    ...
#    selected_funds_for_plot = st.multiselect(...)
#    if selected_funds_for_plot:
#        actual_plot_pivot = ...
#
#        # Use the estimates_log_df that's now from Google Sheets
#        estimated_plot_data = estimates_log_df[estimates_log_df['FundName'].isin(selected_funds_for_plot)].copy()
#        if 'EstimationForDate' in estimated_plot_data.columns: # Ensure col exists
#            estimated_plot_data['EstimationForDate'] = pd.to_datetime(estimated_plot_data['EstimationForDate'])
#        # ... rest of the plotting logic for estimates and combined chart ...


# --- FULL main_app.py LOGIC (incorporating all changes) ---
# (This combines the GSheet logic with the previous structure from main_app_py_v8)

# (Import and Configuration sections as above)

# --- Google Sheets Helper Functions ---
# (init_gspread_client, load_estimates_from_gsheet, save_estimate_to_gsheet as defined above)

gs_client = init_gspread_client()
# Load estimates_log_df from Google Sheets at the start of the session
# This variable will be updated in-session if new estimates are saved.
estimates_log_df_global = load_estimates_from_gsheet(gs_client, GOOGLE_SHEET_NAME, WORKSHEET_NAME)


# --- Global Data Loading (cached) ---
@st.cache_data(ttl=3600) 
def load_all_fund_data(uploaded_file_obj):
    # ... (this function remains the same as in main_app_py_v8, loading price history and holdings) ...
    price_history_df = pd.DataFrame() 
    if uploaded_file_obj is not None:
        price_history_df = load_unit_price_history(uploaded_file_obj)
        if 'Date' in price_history_df.columns:
            price_history_df['Date'] = pd.to_datetime(price_history_df['Date'], errors='coerce')
            price_history_df.dropna(subset=['Date'], inplace=True)
        st.sidebar.success(f"Loaded unit prices from: {uploaded_file_obj.name}")
    else:
        st.sidebar.warning("Please upload a Unit Price History CSV to begin analysis.")
        default_unit_price_file = os.path.join(DATA_DIR, "FutureSaver_UnitPriceHistory-22-05-2025.csv") 
        if os.path.exists(default_unit_price_file):
            price_history_df = load_unit_price_history(default_unit_price_file)
            if 'Date' in price_history_df.columns:
                 price_history_df['Date'] = pd.to_datetime(price_history_df['Date'], errors='coerce')
                 price_history_df.dropna(subset=['Date'], inplace=True)
            st.sidebar.info(f"Using default unit price file from local 'data' folder: {default_unit_price_file}")
        else:
             price_history_df = pd.DataFrame(columns=['Date', 'FundName', 'UnitPrice'])

    all_fund_holdings_data = {}
    combined_keywords = set()
    for fund_spec in HOLDINGS_FILES_INFO:
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

@st.cache_data(ttl=900) 
def fetch_process_news(keywords_list):
    # ... (this function remains the same as in main_app_py_v8) ...
    raw_news = aggregate_news(RSS_FEEDS_CONFIG, keywords_list, days_history=3) 
    analyzed_news = []
    for item in raw_news:
        title_str = str(item.get("title", ""))
        summary_str = str(item.get("summary", ""))
        sentiment = get_sentiment(title_str + " " + summary_str)
        item["sentiment_label"] = sentiment["label"]
        item["sentiment_compound"] = sentiment["compound"]
        item["sentiment_strength"] = classify_sentiment_strength(sentiment["compound"])
        analyzed_news.append(item)
    return analyzed_news

# --- Main App Logic ---
price_data, holdings_data_all_funds, all_keywords = load_all_fund_data(uploaded_unit_price_file)

if price_data.empty and not uploaded_unit_price_file:
    st.error("Please upload a Unit Price History CSV file to start the analysis.")
    st.stop()
elif price_data.empty and uploaded_unit_price_file:
    st.error(f"Failed to process the uploaded unit price file: {uploaded_unit_price_file.name}. Check console for errors, or ensure the file is not empty and correctly formatted.")
    st.stop()

news_items_analyzed = []
if all_keywords:
    news_items_analyzed = fetch_process_news(all_keywords)

st.sidebar.header("Fund Selection")
available_fund_names = sorted([f_info["name"] for f_info in HOLDINGS_FILES_INFO if f_info["name"] in holdings_data_all_funds and not holdings_data_all_funds[f_info["name"]].empty])
if not available_fund_names:
    st.error("No fund data loaded successfully for selection. Check console.")
    st.stop()
selected_fund_for_detail = st.sidebar.selectbox("Choose a fund for detailed analysis:", available_fund_names)

st.header(f"Detailed Analysis for: {selected_fund_for_detail}")

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
            if previous_entry['Date'] < last_price_date:
                previous_price = previous_entry['UnitPrice']
                # ... (actual change display logic as in v8) ...
                previous_price_date_str = previous_entry['Date'].strftime('%d/%m/%Y')
                actual_change_val = last_price - previous_price
                actual_change_percent = (actual_change_val / previous_price) * 100 if previous_price != 0 else 0
                actual_change_str = f"{actual_change_val:+.{UNIT_PRICE_DISPLAY_PRECISION}f}"
                actual_change_percent_str = f"{actual_change_percent:+.4f}%"
                st.metric(label=f"Actual Change from {previous_price_date_str}", value=actual_change_str, delta=f"{actual_change_percent_str}")
            else:
                st.info("Previous day's price not directly before latest; using latest as baseline for estimation.")
                previous_price = last_price 
        else:
            st.info("Not enough historical data to calculate actual day-over-day change.")
            previous_price = last_price 
    else: st.warning(f"No price history found for {selected_fund_for_detail} in the uploaded file.")
else: st.warning("Unit price history data is not available or not loaded.")

current_holdings = holdings_data_all_funds.get(selected_fund_for_detail)
baseline_price_for_estimation = previous_price if previous_price > 0 else last_price 

if current_holdings is not None and not current_holdings.empty and news_items_analyzed and baseline_price_for_estimation > 0 and last_price_date is not None:
    st.subheader(f"Impact Estimation on Unit Price (Estimating for {last_price_date.strftime('%d/%m/%Y')} based on news up to today):")
    with st.spinner("Calculating estimated impact..."):
        # ... (IEM keyword filtering logic as in v8) ...
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

            if last_price_date: 
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
                # ... (Detailed impact display as in v8) ...
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

if not price_data.empty:
    all_fund_names_for_plot = sorted(price_data['FundName'].unique())
    default_selection_for_plot = [selected_fund_for_detail] if selected_fund_for_detail in all_fund_names_for_plot else ([all_fund_names_for_plot[0]] if all_fund_names_for_plot else [])

    selected_funds_for_plot = st.multiselect(
        "Select fund(s) to plot:",
        options=all_fund_names_for_plot,
        default=default_selection_for_plot
    )

    if selected_funds_for_plot:
        actual_plot_data = price_data[price_data['FundName'].isin(selected_funds_for_plot)].copy()
        if not actual_plot_data.empty:
            actual_plot_data['Date'] = pd.to_datetime(actual_plot_data['Date'])
            actual_plot_pivot = actual_plot_data.pivot_table(
                index='Date', columns='FundName', values='UnitPrice'
            ).sort_index()
            actual_plot_pivot.columns = [f"{col}_Actual" for col in actual_plot_pivot.columns]
            
            # Use the globally managed estimates_log_df_global
            estimated_plot_data_source = estimates_log_df_global[estimates_log_df_global['FundName'].isin(selected_funds_for_plot)].copy()
            
            combined_plot_df = actual_plot_pivot
            
            if not estimated_plot_data_source.empty:
                if 'EstimationForDate' in estimated_plot_data_source.columns:
                     estimated_plot_data_source['EstimationForDate'] = pd.to_datetime(estimated_plot_data_source['EstimationForDate'])
                
                estimated_plot_data_agg = estimated_plot_data_source.groupby(['EstimationForDate', 'FundName'])['EstimatedUnitPrice'].mean().reset_index()
                
                estimated_plot_pivot = estimated_plot_data_agg.pivot_table(
                    index='EstimationForDate', columns='FundName', values='EstimatedUnitPrice'
                ).sort_index()
                estimated_plot_pivot.columns = [f"{col}_Estimated" for col in estimated_plot_pivot.columns]
                estimated_plot_pivot.index.name = 'Date' 

                combined_plot_df = pd.merge(actual_plot_pivot, estimated_plot_pivot, on='Date', how='outer').sort_index()
            
            if not combined_plot_df.empty:
                cols_to_plot = []
                for fund_name_plot in selected_funds_for_plot:
                    actual_col_name = f"{fund_name_plot}_Actual"
                    estimated_col_name = f"{fund_name_plot}_Estimated"
                    if actual_col_name in combined_plot_df.columns: cols_to_plot.append(actual_col_name)
                    if estimated_col_name in combined_plot_df.columns: cols_to_plot.append(estimated_col_name)
                
                if cols_to_plot:
                    st.line_chart(combined_plot_df[cols_to_plot])
                else: st.write("No data columns available for plotting.")
            else: st.write("No data for selected fund(s) to plot (actual or estimated).")
        else: st.write("No actual price data for selected fund(s) to plot.")
    else: st.info("Select one or more funds to display their historical prices.")
else: st.warning("Unit price data not loaded. Cannot display historical chart.")

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
