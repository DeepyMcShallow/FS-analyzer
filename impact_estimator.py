# impact_estimator.py
import pandas as pd 

IMPACT_RULES = {
    "Equities": {"Very Positive": 0.030, "Moderate Positive": 0.010, "Neutral": 0.000, "Moderate Negative": -0.010, "Very Negative": -0.030},
    "Property_Specific": {"Very Positive": 0.025, "Moderate Positive": 0.008, "Neutral": 0.000, "Moderate Negative": -0.008, "Very Negative": -0.025},
    "Property_GeneralMarket": {"Very Positive": 0.005, "Moderate Positive": 0.002, "Neutral": 0.000, "Moderate Negative": -0.002, "Very Negative": -0.005},
    "Bonds": {"Very Positive": 0.005, "Moderate Positive": 0.0025, "Neutral": 0.000, "Moderate Negative": -0.0025, "Very Negative": -0.005},
    "Cash": {"Default": 0.000},
    "Term Deposit": {"Default": 0.000},
    "DefaultAsset": {"Very Positive": 0.005, "Moderate Positive": 0.002, "Neutral": 0.000, "Moderate Negative": -0.002, "Very Negative": -0.005}
}

def classify_sentiment_strength_local(compound_score):
    if compound_score >= 0.7: return "Very Positive"
    if 0.05 <= compound_score < 0.7: return "Moderate Positive" 
    if -0.05 < compound_score < 0.05: return "Neutral" 
    if -0.7 < compound_score <= -0.05: return "Moderate Negative"
    if compound_score <= -0.7: return "Very Negative"
    return "Neutral"

def get_impact_percentage_for_asset(asset_class_str, sentiment_strength_str, is_general_market_news=False):
    asset_class_str = str(asset_class_str).lower() 
    ruleset_key = "DefaultAsset" 
    if "equity" in asset_class_str or "share" in asset_class_str: ruleset_key = "Equities"
    elif "property" in asset_class_str: ruleset_key = "Property_GeneralMarket" if is_general_market_news else "Property_Specific"
    elif "bond" in asset_class_str or "fixed income" in asset_class_str or "fixed interest" in asset_class_str : ruleset_key = "Bonds"
    elif "cash" in asset_class_str: ruleset_key = "Cash"
    elif "term deposit" in asset_class_str: ruleset_key = "Term Deposit"
    
    rule_set_to_apply = IMPACT_RULES.get(ruleset_key, IMPACT_RULES["DefaultAsset"])
    impact_value = rule_set_to_apply.get(sentiment_strength_str, 0.0) if "Default" not in rule_set_to_apply else rule_set_to_apply["Default"]
    # print(f"    DEBUG IEM - get_impact_percentage: AC='{asset_class_str}', Strength='{sentiment_strength_str}', General={is_general_market_news}, Ruleset='{ruleset_key}', ImpactVal={impact_value*100:.2f}%")
    return impact_value

def calculate_fund_impact(fund_holdings_df, news_items_with_sentiment, last_known_unit_price):
    # This print is already in the CLI output
    # print(f"\n--- IEM: Calculating impact. Received {len(news_items_with_sentiment)} news items. Last Price: {last_known_unit_price:.4f} ---")
    total_fund_percentage_change = 0.0
    impact_details = []
    
    if fund_holdings_df.empty:
        print("IEM Warning: fund_holdings_df is empty. Cannot calculate impact.")
        return {"total_estimated_fund_percentage_change": 0.0, "estimated_new_unit_price": last_known_unit_price, "impact_details": []}

    holdings_df_copy = fund_holdings_df.copy()
    if "CanonicalHoldingName" in holdings_df_copy.columns:
        holdings_df_copy['_normalized_name'] = holdings_df_copy['CanonicalHoldingName'].astype(str).str.lower().str.strip()
    else: holdings_df_copy['_normalized_name'] = pd.Series(dtype='str')
    
    if "Ticker" in holdings_df_copy.columns:
        holdings_df_copy['_normalized_ticker'] = holdings_df_copy['Ticker'].astype(str).str.lower().str.strip()
    else: holdings_df_copy['_normalized_ticker'] = pd.Series(dtype='str')

    holdings_map_name = {name: row for name, row in holdings_df_copy.set_index("_normalized_name").iterrows() if name and name != 'nan'}
    holdings_map_ticker = {ticker: row for ticker, row in holdings_df_copy.set_index("_normalized_ticker").iterrows() if ticker and ticker != 'nan'}
    
    # print(f"IEM DEBUG: Holdings Map Names (sample): {list(holdings_map_name.keys())[:5]}")
    # print(f"IEM DEBUG: Holdings Map Tickers (sample): {list(holdings_map_ticker.keys())[:5]}")

    news_item_counter = 0
    for news_item in news_items_with_sentiment:
        news_item_counter += 1
        affected_holding_details = None 
        
        sentiment_compound = news_item.get("sentiment_compound", 0.0)
        sentiment_strength = classify_sentiment_strength_local(sentiment_compound) 
        
        news_title_short = news_item.get('title', 'No Title')[:60]
        # This print is already in the CLI output
        # print(f"\n  IEM Processing News Item {news_item_counter}: '{news_title_short}...'")
        # print(f"    Sentiment Compound: {sentiment_compound:.2f}, Strength: {sentiment_strength}")

        # 'matched_keyword' is the one from news_aggregator that got this news item selected.
        # 'display_matched_keyword' in main_app is for display context, might be prefixed with "General: ".
        # For IEM logic, we need the raw keyword that triggered the news selection.
        keyword_for_iem_match = str(news_item.get("matched_keyword", "")).lower().strip() # Use the original matched keyword
        
        if not keyword_for_iem_match or keyword_for_iem_match == '-':
            print(f"    IEM Skipping news item '{news_title_short}' - invalid or empty matched_keyword from aggregator: '{keyword_for_iem_match}'.")
            continue
        # This print is also fine
        # print(f"    IEM Keyword for Matching: '{keyword_for_iem_match}'")

        if keyword_for_iem_match in holdings_map_ticker:
            affected_holding_details = holdings_map_ticker[keyword_for_iem_match]
            # This print is also fine
            # print(f"    IEM MATCHED to Ticker: '{keyword_for_iem_match}' -> Holding: '{affected_holding_details.get('CanonicalHoldingName', 'N/A')}'")
        elif keyword_for_iem_match in holdings_map_name:
            affected_holding_details = holdings_map_name[keyword_for_iem_match]
            # This print is also fine
            # print(f"    IEM MATCHED to Name: '{keyword_for_iem_match}' -> Holding: '{affected_holding_details.get('CanonicalHoldingName', 'N/A')}'")
        
        # General news keywords should be checked against keyword_for_iem_match
        is_general_property_news = any(kw == keyword_for_iem_match for kw in ["property market", "housing price", "construction industry"])
        is_general_interest_rate_news = any(kw == keyword_for_iem_match for kw in ["interest rate", "rba", "monetary policy", "inflation", "cpi"])

        if affected_holding_details is not None: 
            asset_class = str(affected_holding_details.get("AssetClass", "DefaultAsset")).strip()
            holding_weighting_pct = affected_holding_details.get("Weighting", 0.0)
            if pd.isna(holding_weighting_pct): holding_weighting_pct = 0.0
            
            if holding_weighting_pct == 0.0 and sentiment_strength != "Neutral":
                # This print is also fine
                # print(f"    IEM Specific Impact on '{affected_holding_details.get('CanonicalHoldingName')}': Weight is 0.0%, no contribution calculated.")
                pass # Will still add to impact_details for transparency
            
            holding_weighting_decimal = holding_weighting_pct / 100.0
            is_general_news_flag_for_asset = news_item.get("is_general_market_news_type", False) 
            est_impact_on_holding_price = get_impact_percentage_for_asset(asset_class, sentiment_strength, is_general_news_flag_for_asset)
            contribution_to_fund_change = est_impact_on_holding_price * holding_weighting_decimal
            
            if est_impact_on_holding_price != 0.0 and holding_weighting_pct != 0.0 : 
                total_fund_percentage_change += contribution_to_fund_change
                # This print is also fine
                # print(f"    IEM Specific Impact on '{affected_holding_details.get('CanonicalHoldingName')}' (Weight: {holding_weighting_pct}%): AssetImpact={est_impact_on_holding_price*100:.2f}%, FundContrib={contribution_to_fund_change*100:.4f}%")
            elif sentiment_strength != "Neutral" and holding_weighting_pct != 0.0: 
                 # This print is also fine
                 # print(f"    IEM Specific Impact on '{affected_holding_details.get('CanonicalHoldingName')}': Sentiment '{sentiment_strength}', but rule results in 0.0% asset impact.")
                 pass

            impact_details.append({
                "holding_name": affected_holding_details.get("CanonicalHoldingName", "Unknown Specific Holding"), 
                "news_title": news_item["title"], "sentiment_label": sentiment_strength, 
                "sentiment_score": sentiment_compound, "est_impact_on_holding_price_percent": est_impact_on_holding_price * 100,
                "contribution_to_fund_percent_change": contribution_to_fund_change * 100})
        
        elif is_general_property_news: 
            # This print is also fine
            # print(f"    IEM Processing as GENERAL PROPERTY news (keyword: '{keyword_for_iem_match}').")
            applied_to_general = False
            for _, holding_row in holdings_df_copy[holdings_df_copy['AssetClass'].str.lower().str.contains("property", na=False)].iterrows():
                holding_weight_pct = holding_row.get("Weighting", 0.0)
                if pd.isna(holding_weight_pct) or holding_weight_pct == 0.0: continue
                est_impact = get_impact_percentage_for_asset(holding_row["AssetClass"], sentiment_strength, True)
                if est_impact == 0.0 and sentiment_strength != "Neutral": continue
                contribution = est_impact * (holding_weight_pct / 100.0)
                total_fund_percentage_change += contribution
                applied_to_general = True
                # This print is also fine
                # print(f"      IEM General Property Impact on '{holding_row['CanonicalHoldingName']}' (Weight: {holding_weight_pct}%): AssetImpact={est_impact*100:.2f}%, FundContrib={contribution*100:.4f}%")
                impact_details.append({
                    "holding_name": f"General Property Impact ({holding_row['CanonicalHoldingName']})", 
                    "news_title": news_item["title"], "sentiment_label": sentiment_strength, 
                    "sentiment_score": sentiment_compound, "est_impact_on_holding_price_percent": est_impact * 100,
                    "contribution_to_fund_percent_change": contribution * 100})
            # if not applied_to_general: print("      IEM: No property holdings with weight found for this general news.")

        elif is_general_interest_rate_news: 
            # This print is also fine
            # print(f"    IEM Processing as GENERAL INTEREST RATE news (keyword: '{keyword_for_iem_match}').")
            applied_to_general = False
            for _, holding_row in holdings_df_copy[holdings_df_copy['AssetClass'].str.lower().str.contains("bond|fixed income|fixed interest", na=False)].iterrows():
                holding_weight_pct = holding_row.get("Weighting", 0.0)
                if pd.isna(holding_weight_pct) or holding_weight_pct == 0.0: continue
                est_impact = get_impact_percentage_for_asset(holding_row["AssetClass"], sentiment_strength, True)
                if est_impact == 0.0 and sentiment_strength != "Neutral": continue
                contribution = est_impact * (holding_weight_pct / 100.0)
                total_fund_percentage_change += contribution
                applied_to_general = True
                # This print is also fine
                # print(f"      IEM General Rate Impact on '{holding_row['CanonicalHoldingName']}' (Weight: {holding_weight_pct}%): AssetImpact={est_impact*100:.2f}%, FundContrib={contribution*100:.4f}%")
                impact_details.append({
                    "holding_name": f"General Rate Impact ({holding_row['CanonicalHoldingName']})", 
                    "news_title": news_item["title"], "sentiment_label": sentiment_strength, 
                    "sentiment_score": sentiment_compound, "est_impact_on_holding_price_percent": est_impact * 100,
                    "contribution_to_fund_percent_change": contribution * 100})
            # if not applied_to_general: print("      IEM: No bond/fixed income holdings with weight found for this general news.")
        else: 
            # This print is also fine
            # print(f"    IEM Skipping news item for impact calculation: Keyword '{keyword_for_iem_match}' did not match a specific holding and was not classified as general property/rate news needing broad application.")
            pass 
    
    estimated_new_unit_price = last_known_unit_price * (1 + total_fund_percentage_change) if last_known_unit_price > 0 else 0
    # This print is also fine
    # print(f"--- IEM Calculation Complete. Total Fund Change: {total_fund_percentage_change*100:.4f}%, Est. New Price: {estimated_new_unit_price:.4f} ---")
    return {
        "total_estimated_fund_percentage_change": total_fund_percentage_change * 100,
        "estimated_new_unit_price": estimated_new_unit_price,
        "impact_details": impact_details
    }
