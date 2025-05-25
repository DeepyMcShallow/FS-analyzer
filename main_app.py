# main_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import io 

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
ESTIMATES_LOG_FILE = os.path.join(DATA_DIR, "estimated_unit_prices_log.csv")

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
uploaded_unit_price_file = st.sidebar.file_uploader("Upload Latest Unit Price History CSV", type=["csv"])

# --- Helper function to load/initialize estimates log ---
def load_or_initialize_estimates_log():
    if os.path.exists(ESTIMATES_LOG_FILE):
        try:
            df = pd.read_csv(ESTIMATES_LOG_FILE)
            # Ensure correct datetime parsing for existing log file
            df['RunDateTime'] = pd.to_datetime(df['RunDateTime'], errors='coerce')
            df['EstimationForDate'] = pd.to_datetime(df['EstimationForDate'], errors='coerce')
            df.dropna(subset=['RunDateTime', 'EstimationForDate'], inplace=True)
            return df
        except Exception as e:
            print(f"Error loading estimates log: {e}. Initializing new log.")
            pass # Fall through to initialize new
    return pd.DataFrame(columns=['RunDateTime', 'EstimationForDate', 'FundName', 
                                 'EstimatedUnitPrice', 'BaselinePriceUsed', 'TotalEstPercentChange'])

def save_estimate_to_log(log_df, new_estimate_data):
    """Appends a new estimate and saves the log. Prevents duplicate for same fund/estimation_date/run_date."""
    new_row_df = pd.DataFrame([new_estimate_data])
    # Convert to datetime if not already (for comparison)
    new_row_df['RunDateTime'] = pd.to_datetime(new_row_df['RunDateTime'])
    new_row_df['EstimationForDate'] = pd.to_datetime(new_row_df['EstimationForDate'])

    # Simple check to avoid exact duplicate rows if app is re-run quickly without new data
    # A more robust check might consider only latest estimate per EstimationForDate per FundName
    if not log_df.empty:
        # Check if a very similar record (same fund, estimation date, and estimate) exists from today
        # This is a basic check.
        today_str = datetime.now().strftime('%Y-%m-%d')
        is_duplicate = log_df[
            (log_df['FundName'] == new_estimate_data['FundName']) &
            (log_df['EstimationForDate'].dt.strftime('%Y-%m-%d') == new_estimate_data['EstimationForDate'].strftime('%Y-%m-%d')) &
            (log_df['RunDateTime'].dt.strftime('%Y-%m-%d') == today_str) &
            (abs(log_df['EstimatedUnitPrice'] - new_estimate_data['EstimatedUnitPrice']) < 1e-9) # Check if estimate is identical
        ].shape[0] > 0
        if is_duplicate:
            print(f"Skipping save for likely duplicate estimate for {new_estimate_data['FundName']} on {new_estimate_data['EstimationForDate'].strftime('%Y-%m-%d')}")
            return log_df # Return original log

    updated_log_df = pd.concat([log_df, new_row_df], ignore_index=True)
    try:
        updated_log_df.to_csv(ESTIMATES_LOG_FILE, index=False)
        print(f"Successfully saved estimate for {new_estimate_data['FundName']} to log.")
    except Exception as e:
        print(f"Error saving estimates log: {e}")
    return updated_log_df

# Load existing estimates log
estimates_log_df = load_or_initialize_estimates_log()


# --- Global Data Loading (cached) ---
@st.cache_data(ttl=3600) 
def load_all_fund_data(uploaded_file_obj):
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
            st.sidebar.info(f"Using default unit price file: {default_unit_price_file}")
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
elif price_data.empty and uploaded_unit_price_file:
    st.error(f"Failed to process the uploaded unit price file: {uploaded_unit_price_file.name}. Check console for errors, or ensure the file is not empty and correctly formatted.")
    st.stop()

news_items_analyzed = []
if all_keywords:
    news_items_analyzed = fetch_process_news(all_keywords)
else:
    st.warning("No keywords generated from holdings. News fetching skipped. Check holdings files in 'data' folder.")

st.sidebar.header("Fund Selection")
available_fund_names = sorted([f_info["name"] for f_info in HOLDINGS_FILES_INFO if f_info["name"] in holdings_data_all_funds and not holdings_data_all_funds[f_info["name"]].empty])
if not available_fund_names:
    st.error("No fund data loaded successfully for selection. Check console for data loading errors for holdings files.")
    st.stop()
selected_fund_for_detail = st.sidebar.selectbox("Choose a fund for detailed analysis:", available_fund_names)

st.header(f"Detailed Analysis for: {selected_fund_for_detail}")

last_price = 0.0
previous_price = 0.0
last_price_date = None # Store the actual datetime object
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
    else: 
        st.warning(f"No price history found for {selected_fund_for_detail} in the uploaded file.")
else: 
    st.warning("Unit price history data is not available or not loaded.")

current_holdings = holdings_data_all_funds.get(selected_fund_for_detail)
baseline_price_for_estimation = previous_price if previous_price > 0 else last_price 

if current_holdings is not None and not current_holdings.empty and news_items_analyzed and baseline_price_for_estimation > 0 and last_price_date is not None:
    st.subheader(f"Impact Estimation on Unit Price (Estimating for {last_price_date.strftime('%d/%m/%Y')} based on news up to today):")
    with st.spinner("Calculating estimated impact..."):
        # ... (IEM keyword filtering logic remains the same as main_app_py_v8) ...
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

            # --- Save the estimate to log ---
            if last_price_date: # Ensure we have a valid date for the estimation
                new_estimate_data = {
                    'RunDateTime': datetime.now(),
                    'EstimationForDate': last_price_date, # The date this estimate applies to
                    'FundName': selected_fund_for_detail,
                    'EstimatedUnitPrice': estimated_unit_price,
                    'BaselinePriceUsed': baseline_price_for_estimation,
                    'TotalEstPercentChange': total_est_percent_change
                }
                estimates_log_df = save_estimate_to_log(estimates_log_df, new_estimate_data)
                # No need to assign back to global here, save_estimate_to_log updates the file.
                # For immediate reflection in plotting if needed, could re-load or update in-memory df.
                # For simplicity, chart will use data from log file loaded at start or after save.

            with st.expander("Show Detailed Impact Calculations"):
                # ... (Detailed impact display remains the same) ...
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

# --- Historical Unit Price Chart (Actual vs. Estimated) ---
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
        # Prepare Actual Prices Data
        actual_plot_data = price_data[price_data['FundName'].isin(selected_funds_for_plot)].copy()
        if not actual_plot_data.empty:
            actual_plot_data['Date'] = pd.to_datetime(actual_plot_data['Date'])
            actual_plot_pivot = actual_plot_data.pivot_table(
                index='Date', columns='FundName', values='UnitPrice'
            ).sort_index()
            actual_plot_pivot.columns = [f"{col}_Actual" for col in actual_plot_pivot.columns] # Rename for legend

            # Prepare Estimated Prices Data (from log)
            # Ensure estimates_log_df is up-to-date if new estimates were just saved
            # For simplicity, we use the version loaded at app start or after the latest save
            # A more reactive way would be to pass estimates_log_df as a mutable object or use session state.
            # For now, let's re-load it if a save just happened, or use the one in memory.
            # We need to ensure 'EstimationForDate' is datetime
            current_estimates_log_df = load_or_initialize_estimates_log() # Get the latest from file
            if 'EstimationForDate' in current_estimates_log_df.columns:
                 current_estimates_log_df['EstimationForDate'] = pd.to_datetime(current_estimates_log_df['EstimationForDate'])


            estimated_plot_data = current_estimates_log_df[current_estimates_log_df['FundName'].isin(selected_funds_for_plot)].copy()
            
            combined_plot_df = actual_plot_pivot
            
            if not estimated_plot_data.empty:
                # To plot estimates for the *same day* as actuals, we use 'EstimationForDate'
                # We might have multiple estimates for the same day if app run multiple times.
                # For plotting, let's take the average or latest estimate per day.
                estimated_plot_data_agg = estimated_plot_data.groupby(['EstimationForDate', 'FundName'])['EstimatedUnitPrice'].mean().reset_index()
                
                estimated_plot_pivot = estimated_plot_data_agg.pivot_table(
                    index='EstimationForDate', columns='FundName', values='EstimatedUnitPrice'
                ).sort_index()
                estimated_plot_pivot.columns = [f"{col}_Estimated" for col in estimated_plot_pivot.columns]
                estimated_plot_pivot.index.name = 'Date' # Align index name for merge

                # Merge actual and estimated data
                combined_plot_df = pd.merge(actual_plot_pivot, estimated_plot_pivot, on='Date', how='outer').sort_index()
            
            if not combined_plot_df.empty:
                # Select only columns for currently selected funds for plotting
                cols_to_plot = []
                for fund_name_plot in selected_funds_for_plot:
                    if f"{fund_name_plot}_Actual" in combined_plot_df.columns:
                        cols_to_plot.append(f"{fund_name_plot}_Actual")
                    if f"{fund_name_plot}_Estimated" in combined_plot_df.columns:
                        cols_to_plot.append(f"{fund_name_plot}_Estimated")
                
                if cols_to_plot:
                    st.line_chart(combined_plot_df[cols_to_plot])
                else:
                    st.write("No data columns available for plotting after merging.")
            else:
                st.write("No data available for the selected fund(s) to plot (actual or estimated).")
        else:
            st.write("No actual price data available for the selected fund(s) to plot.")
    else:
        st.info("Select one or more funds from the dropdown above to display their historical unit prices.")
else:
    st.warning("Unit price data not loaded. Cannot display historical chart.")

# --- Display Relevant News ---
if current_holdings is not None and not current_holdings.empty:
    # ... (News display section remains the same as main_app_py_v8) ...
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
