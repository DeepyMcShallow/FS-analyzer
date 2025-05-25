# main_app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone # Ensure timezone is imported
import os
import io 

# --- Import your custom modules ---
try:
    from data_loader import load_unit_price_history, load_fund_holdings
    from news_aggregator import aggregate_news, RSS_FEEDS_CONFIG 
    from sentiment_analyzer import get_sentiment, classify_sentiment_strength # This should be the corrected version
    from impact_estimator import calculate_fund_impact
except ImportError as e:
    st.error(f"CRITICAL ERROR: Could not import one or more custom modules: {e}. ")
    st.stop()

# --- Configuration ---
DATA_DIR = "data" 
UNIT_PRICE_DISPLAY_PRECISION = 6 
ESTIMATES_LOG_FILE = os.path.join(DATA_DIR, "estimated_unit_prices_log.csv") # Local CSV log

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
st.caption(f"Report Generated: {datetime.now(timezone(timedelta(hours=10))).strftime('%A, %d %B %Y, %I:%M %p AEST')}")


st.sidebar.header("Data Upload")
uploaded_unit_price_file = st.sidebar.file_uploader("Upload Latest Unit Price History CSV", type=["csv"])

# --- Helper function to load/initialize estimates log (from local CSV) ---
def load_or_initialize_estimates_log_local():
    expected_cols = ['RunDateTime', 'EstimationForDate', 'FundName', 
                     'EstimatedUnitPrice', 'BaselinePriceUsed', 'TotalEstPercentChange']
    if os.path.exists(ESTIMATES_LOG_FILE):
        try:
            df = pd.read_csv(ESTIMATES_LOG_FILE)
            if 'RunDateTime' in df.columns:
                 df['RunDateTime'] = pd.to_datetime(df['RunDateTime'], errors='coerce')
            if 'EstimationForDate' in df.columns:
                 df['EstimationForDate'] = pd.to_datetime(df['EstimationForDate'], errors='coerce')
            df.dropna(subset=['RunDateTime', 'EstimationForDate', 'FundName'], how='any', inplace=True) # Drop if key dates are bad
            # Ensure all expected columns exist
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = pd.NA # Or appropriate default
            return df[expected_cols] # Return in expected order
        except Exception as e:
            print(f"Error loading local estimates log '{ESTIMATES_LOG_FILE}': {e}. Initializing new log.")
    return pd.DataFrame(columns=expected_cols)

def save_estimate_to_log_local(log_df, new_estimate_data):
    """Appends a new estimate and saves the log to local CSV."""
    new_row_df = pd.DataFrame([new_estimate_data])
    if 'RunDateTime' in new_row_df.columns:
        new_row_df['RunDateTime'] = pd.to_datetime(new_row_df['RunDateTime'])
    if 'EstimationForDate' in new_row_df.columns:
        new_row_df['EstimationForDate'] = pd.to_datetime(new_row_df['EstimationForDate'])

    # Basic duplicate check for this session (can be improved)
    # For local CSV, overwriting or just appending might be fine.
    # This check prevents exact re-runs in quick succession from bloating the log too much.
    if not log_df.empty and not new_row_df.empty:
        # Check against last entry for same fund and estimation date
        last_entry_for_fund_date = log_df[
            (log_df['FundName'] == new_estimate_data['FundName']) &
            (log_df['EstimationForDate'] == new_estimate_data['EstimationForDate'])
        ].tail(1)
        if not last_entry_for_fund_date.empty:
            if abs(last_entry_for_fund_date['EstimatedUnitPrice'].iloc[0] - new_estimate_data['EstimatedUnitPrice']) < 1e-9:
                print(f"Skipping save to local log for likely duplicate estimate: {new_estimate_data['FundName']}")
                return log_df


    updated_log_df = pd.concat([log_df, new_row_df], ignore_index=True)
    try:
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        updated_log_df.to_csv(ESTIMATES_LOG_FILE, index=False)
        print(f"Successfully saved estimate for {new_estimate_data['FundName']} to local log: {ESTIMATES_LOG_FILE}")
    except Exception as e:
        print(f"Error saving estimates to local log '{ESTIMATES_LOG_FILE}': {e}")
    return updated_log_df

# Load existing estimates log (local version)
estimates_log_df_global = load_or_initialize_estimates_log_local()


# --- Global Data Loading (cached) ---
@st.cache_data(ttl=3600) 
def load_all_fund_data(uploaded_file_obj_param):
    # ... (this function is the same as in main_app_py_v12/v13) ...
    price_history_df = pd.DataFrame() 
    if uploaded_file_obj_param is not None:
        price_history_df = load_unit_price_history(uploaded_file_obj_param)
        if 'Date' in price_history_df.columns:
            price_history_df['Date'] = pd.to_datetime(price_history_df['Date'], errors='coerce')
            price_history_df.dropna(subset=['Date'], inplace=True)
    else:
        default_unit_price_file = os.path.join(DATA_DIR, "FutureSaver_UnitPriceHistory-22-05-2025.csv") 
        if os.path.exists(default_unit_price_file):
            price_history_df = load_unit_price_history(default_unit_price_file)
            if 'Date' in price_history_df.columns:
                 price_history_df['Date'] = pd.to_datetime(price_history_df['Date'], errors='coerce')
                 price_history_df.dropna(subset=['Date'], inplace=True)
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
    print(f"--- MainApp: Total unique keywords for news aggregation: {len(combined_keywords)} ---") 
    return price_history_df, all_fund_holdings_data, list(combined_keywords)

@st.cache_data(ttl=900) 
def fetch_process_news(keywords_list): 
    # ... (this function is the same as in main_app_py_v12/v13) ...
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

# --- Main App UI and Logic Starts Here ---
# (UI setup as in main_app_py_v12/v13)
# ...

price_data, holdings_data_all_funds, all_keywords = load_all_fund_data(uploaded_unit_price_file)

if price_data.empty and not uploaded_unit_price_file:
    st.sidebar.error("Please upload a Unit Price History CSV to start.")
elif price_data.empty and uploaded_unit_price_file:
    st.error(f"Failed to process the uploaded unit price file: {uploaded_unit_price_file.name}. Check console.")

news_items_analyzed = []
if all_keywords and not any(df.empty for df in holdings_data_all_funds.values()): 
    news_items_analyzed = fetch_process_news(all_keywords)
elif not all_keywords:
    st.warning("No keywords generated (holdings might not have loaded). News fetching skipped.")

st.sidebar.header("Fund Selection")
available_fund_names = sorted([f_info["name"] for f_info in HOLDINGS_FILES_INFO if f_info["name"] in holdings_data_all_funds and not holdings_data_all_funds[f_info["name"]].empty])

if not available_fund_names:
    st.error("No fund data loaded successfully for selection. Check data files and console logs.")
    st.stop() 
selected_fund_for_detail = st.sidebar.selectbox("Choose a fund for detailed analysis:", available_fund_names)

st.header(f"Detailed Analysis for: {selected_fund_for_detail}")

# --- Display Latest Unit Price & ACTUAL Day-over-Day Change ---
# (This section remains the same as main_app_py_v12/v13) ...
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

# --- Impact Estimation Display & Saving (to LOCAL CSV) ---
current_holdings = holdings_data_all_funds.get(selected_fund_for_detail)
baseline_price_for_estimation = previous_price if previous_price > 0 else last_price 

if current_holdings is not None and not current_holdings.empty and news_items_analyzed and baseline_price_for_estimation > 0 and last_price_date is not None:
    st.subheader(f"Impact Estimation on Unit Price (Estimating for {last_price_date.strftime('%d/%m/%Y')} based on news up to today):")
    with st.spinner("Calculating estimated impact..."):
        # ... (IEM keyword filtering logic as in v12/v13) ...
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
                # Call the local CSV save function
                estimates_log_df_global = save_estimate_to_log_local(estimates_log_df_global, new_estimate_data)
                
            with st.expander("Show Detailed Impact Calculations"):
                # ... (Detailed impact display as in v12/v13) ...
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

# ... (Rest of the app layout: Top 5 Holdings, Historical Chart, Relevant News Display as in main_app_py_v12/v13, 
#      making sure to use `estimates_log_df_global` for plotting which now comes from local CSV) ...
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

st.markdown("---")
st.header("Historical Unit Price Chart (Actual vs. Estimated)")
if not price_data.empty : 
    all_fund_names_for_plot = sorted(list(price_data['FundName'].unique())) 
    if not estimates_log_df_global.empty and 'FundName' in estimates_log_df_global.columns:
         all_fund_names_for_plot = sorted(list(set(all_fund_names_for_plot) | set(estimates_log_df_global['FundName'].unique())))
    
    default_selection_for_plot = [selected_fund_for_detail] if selected_fund_for_detail in all_fund_names_for_plot else ([all_fund_names_for_plot[0]] if all_fund_names_for_plot else [])

    selected_funds_for_plot = st.multiselect(
        "Select fund(s) to plot:", options=all_fund_names_for_plot, default=default_selection_for_plot
    )
    if selected_funds_for_plot:
        actual_plot_data = price_data[price_data['FundName'].isin(selected_funds_for_plot)].copy()
        actual_plot_pivot = pd.DataFrame()
        if not actual_plot_data.empty:
            actual_plot_data['Date'] = pd.to_datetime(actual_plot_data['Date'])
            actual_plot_pivot = actual_plot_data.pivot_table(
                index='Date', columns='FundName', values='UnitPrice'
            ).sort_index()
            actual_plot_pivot.columns = [f"{col}_Actual" for col in actual_plot_pivot.columns]
            
        estimated_plot_pivot = pd.DataFrame()
        # Use the estimates_log_df_global which is now from local CSV
        if not estimates_log_df_global.empty and 'FundName' in estimates_log_df_global.columns:
            estimated_plot_data_source = estimates_log_df_global[estimates_log_df_global['FundName'].isin(selected_funds_for_plot)].copy()
            if not estimated_plot_data_source.empty:
                if 'EstimationForDate' in estimated_plot_data_source.columns:
                     estimated_plot_data_source['EstimationForDate'] = pd.to_datetime(estimated_plot_data_source['EstimationForDate'])
                
                estimated_plot_data_agg = estimated_plot_data_source.sort_values('RunDateTime').groupby(
                    ['EstimationForDate', 'FundName'], as_index=False
                ).last() 
                
                if not estimated_plot_data_agg.empty and 'EstimatedUnitPrice' in estimated_plot_data_agg.columns :
                    estimated_plot_pivot = estimated_plot_data_agg.pivot_table(
                        index='EstimationForDate', columns='FundName', values='EstimatedUnitPrice'
                    ).sort_index()
                    estimated_plot_pivot.columns = [f"{col}_Estimated" for col in estimated_plot_pivot.columns]
                    estimated_plot_pivot.index.name = 'Date' 

        if not actual_plot_pivot.empty and not estimated_plot_pivot.empty:
            combined_plot_df = pd.merge(actual_plot_pivot, estimated_plot_pivot, on='Date', how='outer').sort_index()
        elif not actual_plot_pivot.empty: combined_plot_df = actual_plot_pivot
        elif not estimated_plot_pivot.empty: combined_plot_df = estimated_plot_pivot
        else: combined_plot_df = pd.DataFrame()
        
        if not combined_plot_df.empty:
            cols_to_plot = []
            for fund_name_plot in selected_funds_for_plot:
                if f"{fund_name_plot}_Actual" in combined_plot_df.columns: cols_to_plot.append(f"{fund_name_plot}_Actual")
                if f"{fund_name_plot}_Estimated" in combined_plot_df.columns: cols_to_plot.append(f"{fund_name_plot}_Estimated")
            if cols_to_plot: st.line_chart(combined_plot_df[cols_to_plot])
            else: st.write("No data columns for plotting for selected fund(s).")
        else: st.write("No actual or estimated data to plot for selected fund(s).")
    else: st.info("Select fund(s) to display historical prices.")
elif price_data.empty:
    st.warning("Unit price data not loaded. Cannot display historical chart.")
else: 
    st.info("No estimates logged yet to display on historical chart. Actuals can still be plotted.")
    all_fund_names_for_plot = sorted(list(price_data['FundName'].unique()))
    default_selection_for_plot = [selected_fund_for_detail] if selected_fund_for_detail in all_fund_names_for_plot else ([all_fund_names_for_plot[0]] if all_fund_names_for_plot else [])
    selected_funds_for_plot = st.multiselect(
        "Select fund(s) to plot (Actuals Only):", options=all_fund_names_for_plot, default=default_selection_for_plot
    )
    if selected_funds_for_plot:
        actual_plot_data = price_data[price_data['FundName'].isin(selected_funds_for_plot)].copy()
        if not actual_plot_data.empty:
            actual_plot_data['Date'] = pd.to_datetime(actual_plot_data['Date'])
            actual_plot_pivot = actual_plot_data.pivot_table(index='Date', columns='FundName', values='UnitPrice').sort_index()
            st.line_chart(actual_plot_pivot)

if current_holdings is not None and not current_holdings.empty:
    # ... (News display section as in v12/v13) ...
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

