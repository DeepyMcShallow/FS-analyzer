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
# Define the precision for displaying unit prices
UNIT_PRICE_DISPLAY_PRECISION = 6 # Changed from 4 to 6

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
st.caption(f"Report Generated: {datetime.now().strftime('%A, %d %B %Y, %I:%M %p %Z')}")

st.sidebar.header("Data Upload")
uploaded_unit_price_file = st.sidebar.file_uploader("Upload Latest Unit Price History CSV", type=["csv"])

# --- Global Data Loading (cached) ---
@st.cache_data(ttl=3600) 
def load_all_fund_data(uploaded_file_obj):
    price_history_df = pd.DataFrame() 
    if uploaded_file_obj is not None:
        price_history_df = load_unit_price_history(uploaded_file_obj)
        st.sidebar.success(f"Loaded unit prices from: {uploaded_file_obj.name}")
    else:
        st.sidebar.warning("Please upload a Unit Price History CSV to begin analysis.")
        default_unit_price_file = os.path.join(DATA_DIR, "FutureSaver_UnitPriceHistory-22-05-2025.csv") # Example, ensure this exists if used
        if os.path.exists(default_unit_price_file):
            price_history_df = load_unit_price_history(default_unit_price_file)
            st.sidebar.info(f"Using default unit price file: {default_unit_price_file}")
        else:
             price_history_df = pd.DataFrame(columns=['Date', 'FundName', 'UnitPrice'])

    all_fund_holdings_data = {}
    combined_keywords = set()
    # print("\n--- MainApp: Loading All Fund Holdings & Generating Keywords ---") # CLI
    for fund_spec in HOLDINGS_FILES_INFO:
        holdings_file_path = os.path.join(DATA_DIR, fund_spec["file"])
        if not os.path.exists(holdings_file_path):
            print(f"Holdings file not found: {holdings_file_path} for fund {fund_spec['name']}") # CLI
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
    # print(f"--- MainApp: Total unique keywords for news aggregation: {len(combined_keywords)} ---") # CLI
    return price_history_df, all_fund_holdings_data, list(combined_keywords)

@st.cache_data(ttl=900) 
def fetch_process_news(keywords_list):
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
    st.error(f"Failed to process the uploaded unit price file: {uploaded_unit_price_file.name}. Check console for errors.")
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
selected_fund = st.sidebar.selectbox("Choose a fund to analyze:", available_fund_names)

st.header(f"Detailed Analysis for: {selected_fund}")

# --- Display Latest Unit Price & ACTUAL Day-over-Day Change ---
last_price = 0.0
previous_price = 0.0
last_price_date_str = "N/A"
actual_change_str = "N/A"
actual_change_percent_str = "N/A"

if not price_data.empty:
    fund_price_history = price_data[price_data['FundName'] == selected_fund].sort_values(by='Date', ascending=False)
    if not fund_price_history.empty:
        latest_entry = fund_price_history.iloc[0]
        last_price = latest_entry['UnitPrice']
        last_price_date_str = latest_entry['Date'].strftime('%d/%m/%Y')
        # Use the UNIT_PRICE_DISPLAY_PRECISION constant for formatting
        st.subheader(f"Latest Actual Unit Price ({last_price_date_str}): {last_price:.{UNIT_PRICE_DISPLAY_PRECISION}f}")

        if len(fund_price_history) > 1:
            previous_entry = fund_price_history.iloc[1]
            previous_price = previous_entry['UnitPrice']
            previous_price_date_str = previous_entry['Date'].strftime('%d/%m/%Y')
            actual_change_val = last_price - previous_price
            actual_change_percent = (actual_change_val / previous_price) * 100 if previous_price != 0 else 0
            
            # Use the UNIT_PRICE_DISPLAY_PRECISION constant for formatting
            actual_change_str = f"{actual_change_val:+.{UNIT_PRICE_DISPLAY_PRECISION}f}"
            actual_change_percent_str = f"{actual_change_percent:+.4f}%" # Percentage change can stay at 4dp
            
            st.metric(label=f"Actual Change from {previous_price_date_str}", value=actual_change_str, delta=f"{actual_change_percent_str}")
        else:
            st.info("Not enough historical data in the uploaded file to calculate actual day-over-day change for this fund.")
    else: 
        st.warning(f"No price history found for {selected_fund} in the uploaded file.")
else: 
    st.warning("Unit price history data is not available or not loaded.")

current_holdings = holdings_data_all_funds.get(selected_fund)
baseline_price_for_estimation = previous_price if previous_price > 0 else last_price 

if current_holdings is not None and not current_holdings.empty and news_items_analyzed and baseline_price_for_estimation > 0:
    st.subheader("Impact Estimation on Unit Price (Estimating for Today):")
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
            # Use the UNIT_PRICE_DISPLAY_PRECISION constant for formatting
            st.metric("Total Est. % Change from News", f"{iem_result.get('total_estimated_fund_percentage_change',0.0):.4f}%") # Percentage change can stay 4dp
            st.metric("Estimated New Unit Price (for today)", f"{iem_result.get('estimated_new_unit_price', baseline_price_for_estimation):.{UNIT_PRICE_DISPLAY_PRECISION}f}")

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

if current_holdings is not None and not current_holdings.empty:
    st.subheader("Top 5 Holdings:")
    if "Weighting" in current_holdings.columns:
        st.dataframe(current_holdings.nlargest(5, 'Weighting')[["CanonicalHoldingName", "Ticker", "AssetClass", "Weighting", "Currency"]])
    else:
        st.warning("Top holdings display issue: 'Weighting' column missing.")
        cols_to_show = [col for col in ["CanonicalHoldingName", "Ticker", "AssetClass", "Currency"] if col in current_holdings.columns]
        if cols_to_show: st.dataframe(current_holdings.head()[cols_to_show])
else:
    st.warning(f"No holdings data loaded for {selected_fund}.")

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

