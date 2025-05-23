# news_aggregator.py
import feedparser
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from datetime import datetime, timedelta, timezone
import time
from dateutil import parser as date_parser 

PROCESSED_ARTICLE_LINKS_SESSION = set() 

# Updated and more reliable RSS feed list
RSS_FEEDS_CONFIG = {
    "Investing.com AU General": "https://au.investing.com/rss/news_25.rss",
    "Investing.com US Market News": "https://www.investing.com/rss/news_1.rss", 
    "Investing.com World News": "https://www.investing.com/rss/news_14.rss",
    "Reuters Business (Global via Feedburner)": "https://feeds.feedburner.com/reuters/businessNews",
    "Reuters World News (Global via Feedburner)": "https://feeds.feedburner.com/Reuters/worldNews",
    "Reuters US Domestic News (via Feedburner)": "https://feeds.feedburner.com/Reuters/domesticNews",
    "ABC News Business (AU)": "https://www.abc.net.au/news/feed/51892/rss.xml",
    "RBA Media Releases (AU - User Provided)": "https://www.rba.gov.au/rss/rss-cb-media-releases.xml", 
    "Yahoo Finance World Market News": "https://finance.yahoo.com/news/rssindex", 
    "Yahoo Finance Top Stories": "https://finance.yahoo.com/rss/topstories"
}


def fetch_news_from_feed(feed_url, feed_name="Unknown Feed"):
    """Fetches news items from a single RSS feed with improved error handling and date parsing."""
    news_items = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 FutureSaverApp/1.0'
    }
    
    try:
        # print(f"Fetching news from {feed_name} ({feed_url})...") 
        response = requests.get(feed_url, headers=headers, timeout=25)
        response.raise_for_status() 
        feed = feedparser.parse(response.content)

        if feed.bozo:
            bozo_exception = getattr(feed, 'bozo_exception', 'Unknown parsing error')
            if isinstance(bozo_exception, Exception):
                 print(f"Warning: feedparser issue with feed {feed_name}. Error: {bozo_exception}")

        for entry in feed.entries:
            link = getattr(entry, 'link', None)
            if not link or link in PROCESSED_ARTICLE_LINKS_SESSION:
                continue

            title = getattr(entry, 'title', "No Title Provided")
            
            published_parsed_dt = None
            date_struct_sources = [getattr(entry, 'published_parsed', None), getattr(entry, 'updated_parsed', None)]
            for date_struct in date_struct_sources:
                if date_struct:
                    try:
                        dt_naive = datetime.fromtimestamp(time.mktime(date_struct))
                        published_parsed_dt = dt_naive.replace(tzinfo=timezone.utc) 
                        break
                    except Exception:
                        pass
            
            if not published_parsed_dt:
                date_string_sources = [getattr(entry, 'published', None), getattr(entry, 'updated', None)]
                for date_str in date_string_sources:
                    if date_str:
                        try:
                            dt_temp = date_parser.parse(date_str)
                            published_parsed_dt = dt_temp.astimezone(timezone.utc) if dt_temp.tzinfo else dt_temp.replace(tzinfo=timezone.utc)
                            break 
                        except Exception:
                            pass
            
            if not published_parsed_dt:
                published_parsed_dt = datetime.now(timezone.utc)

            summary_raw = getattr(entry, 'summary', title)
            summary = summary_raw
            if '<' in summary_raw and '>' in summary_raw: 
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(summary_raw, "html.parser")
                    summary = soup.get_text(separator=" ", strip=True)
                except ImportError: 
                    import re
                    summary = re.sub(r'<[^>]+>', '', summary_raw)

            news_items.append({
                "title": title, "link": link, "published_datetime": published_parsed_dt,
                "published_str": published_parsed_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "summary": summary, "source_feed_name": feed_name, "source_feed_url": feed_url
            })
            PROCESSED_ARTICLE_LINKS_SESSION.add(link)
            
    except Timeout: print(f"Timeout error fetching feed {feed_name} ({feed_url}).")
    except ConnectionError: print(f"Connection error fetching feed {feed_name} ({feed_url}). Check network or URL.")
    except RequestException as e: print(f"Request error fetching feed {feed_name} ({feed_url}): {e}")
    except Exception as e: print(f"Unexpected error processing feed {feed_name} ({feed_url}): {e}")
    return news_items

def aggregate_news(rss_feed_dict, relevant_keywords_list, days_history=3):
    """Aggregates news from multiple feeds and filters by keywords and date."""
    all_fetched_news = []
    PROCESSED_ARTICLE_LINKS_SESSION.clear()

    for feed_name, feed_url in rss_feed_dict.items():
        all_fetched_news.extend(fetch_news_from_feed(feed_url, feed_name))
        time.sleep(0.25)
    
    relevant_news = []
    keywords_lower = [str(kw).lower().strip() for kw in relevant_keywords_list if kw and isinstance(kw, str) and str(kw).strip() and str(kw).strip() != '-']
    
    if not keywords_lower:
        print("Warning: No valid keywords provided for news aggregation filtering.")
        return []

    date_threshold = datetime.now(timezone.utc) - timedelta(days=days_history)

    for item in all_fetched_news:
        item_dt = item["published_datetime"]
        if item_dt.tzinfo is None: 
            item_dt = item_dt.replace(tzinfo=timezone.utc)
        if item_dt < date_threshold: continue
            
        text_to_search = (item["title"] + " " + item["summary"]).lower()
        matched_kws_for_item = set()
        for keyword in keywords_lower: 
            if keyword in text_to_search: 
                matched_kws_for_item.add(keyword)
        
        if matched_kws_for_item:
            item["matched_keywords_list"] = list(matched_kws_for_item)
            item["matched_keyword"] = list(matched_kws_for_item)[0] 
            relevant_news.append(item)
    
    relevant_news.sort(key=lambda x: x["published_datetime"], reverse=True)
    return relevant_news

# (Keep the if __name__ == '__main__': block from v4 for testing news_aggregator.py if desired)
# If you use BeautifulSoup for summary cleaning, add 'beautifulsoup4' to requirements.txt
