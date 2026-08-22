import concurrent.futures
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MASTER_SITEMAP = "https://www.brahmakumaris.com/sitemap_index.xml"
SIMILARITY_THRESHOLD = 0.90  # 90%+ match flagged as duplicate
MAX_WORKERS = 5              # Low worker count protects live server performance
REQUEST_DELAY = 0.1          # Polite pause (in seconds) between fetches
OUTPUT_EXCEL = "sitemap_duplicates_report.xlsx"
# ==============================================================================


def get_all_sitemap_urls(master_url):
    """Fetches all URLs across all child sitemaps in the master index."""
    print(f"-> Accessing Master Sitemap Index: {master_url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }
    all_urls = []

    try:
        res = requests.get(master_url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, "xml")

        child_sitemaps = [
            loc.text.strip()
            for loc in soup.find_all("loc")
            if loc.text.endswith(".xml")
        ]
        print(f"Found {len(child_sitemaps)} child sitemaps.")

        for idx, sitemap in enumerate(child_sitemaps, 1):
            print(f"   [{idx}/{len(child_sitemaps)}] Extracting links: {sitemap}")
            try:
                c_res = requests.get(sitemap, headers=headers, timeout=15)
                c_soup = BeautifulSoup(c_res.content, "xml")

                urls = [
                    loc.text.strip()
                    for loc in c_soup.find_all("loc")
                    if loc.text
                    and not loc.text.endswith(
                        (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif")
                    )
                ]
                all_urls.extend(urls)
            except Exception as err:
                print(f"   ⚠️ Could not read {sitemap}: {err}")

    except Exception as e:
        print(f"❌ Error reading master sitemap: {e}")

    unique_urls = list(set(all_urls))
    print(f"\nTotal Unique Webpage URLs Discovered: {len(unique_urls)}")
    return unique_urls


def fetch_page_text(url):
    """Scrapes clean main body text from a single URL safely."""
    time.sleep(REQUEST_DELAY)  # Server polite pause
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return url, ""

        soup = BeautifulSoup(res.text, "html.parser")

        # Strip layout, navigation, script, and footer noise
        for elem in soup(["script", "style", "nav", "footer", "header", "svg", "form"]):
            elem.extract()

        clean_text = " ".join(soup.get_text().split())
        return url, clean_text
    except Exception:
        return url, ""


def choose_primary_url(url_a, text_a, url_b, text_b):
    """
    Applies SEO canonical guidelines to decide which URL to Keep vs Remove:
    1. Check for copy tags in URL slug (-copy, -2, temp, draft)
    2. Prefer URL with richer text content
    3. Prefer cleaner/shorter URL path
    """
    a_is_copy = any(tag in url_a.lower() for tag in ["-copy", "-2", "-3", "temp-", "draft"])
    b_is_copy = any(tag in url_b.lower() for tag in ["-copy", "-2", "-3", "temp-", "draft"])

    if a_is_copy and not b_is_copy:
        return url_b, url_a
    if b_is_copy and not a_is_copy:
        return url_a, url_b

    len_a, len_b = len(text_a.split()), len(text_b.split())
    if abs(len_a - len_b) > 50:
        return (url_a, url_b) if len_a > len_b else (url_b, url_a)

    if len(url_a) != len(url_b):
        return (url_a, url_b) if len(url_a) < len(url_b) else (url_b, url_a)

    return url_a, url_b


def run_master_sitemap_analysis():
    start_time = time.time()

    # 1. Fetch URLs across all sitemaps
    urls = get_all_sitemap_urls(MASTER_SITEMAP)
    total_urls = len(urls)

    if not urls:
        print("❌ No URLs found. Exiting.")
        return

# 2. Parallel Scraping with Local Caching
    import json
    import os

    CACHE_FILE = "cached_pages.json"

    if os.path.exists(CACHE_FILE):
        print(f"\nStep 1: Found cache file! Loading pages from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            pages_data = json.load(f)
    else:
        print(f"\nStep 1: Scraping text content safely using {MAX_WORKERS} parallel workers...")
        pages_data = {}
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(fetch_page_text, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url, text = future.result()
                completed += 1

                if len(text) > 150:
                    pages_data[url] = text

                if completed % 200 == 0 or completed == total_urls:
                    print(f"   Progress: Scraped [{completed}/{total_urls}] pages...")

        # Save scraped data locally so future runs execute in seconds
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=2)

    valid_urls = list(pages_data.keys())
    texts = list(pages_data.values())
    print(f"\nSuccessfully extracted content from {len(valid_urls)} pages.")

    # 3. Local Machine Embedding Calculation
    print("\nStep 2: Loading embedding model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Computing semantic embeddings (processed locally on your PC)...")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    # 4. Matrix Similarity Calculation
    print("\nStep 3: Calculating cross-sitemap similarity matrix...")
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # 5. Build Deduplicated Decision Report
    print("Filtering duplicate pages matching threshold >= 90%...")
    report_data = []
    flagged_as_duplicate = set()

    # Sort matches by highest similarity first
    for i in range(len(valid_urls)):
        url_a = valid_urls[i]
        if url_a in flagged_as_duplicate:
            continue

        for j in range(i + 1, len(valid_urls)):
            url_b = valid_urls[j]
            if url_b in flagged_as_duplicate:
                continue

            score = cosine_scores[i][j].item()

            if score >= SIMILARITY_THRESHOLD:
                text_a = pages_data[url_a]
                text_b = pages_data[url_b]

                keep_url, remove_url = choose_primary_url(url_a, text_a, url_b, text_b)
                flagged_as_duplicate.add(remove_url)

                report_data.append({
                    "Match Score": f"{round(score * 100, 2)}%",
                    "Similarity Value": score,
                    "Primary Page (KEEP)": keep_url,
                    "Duplicate Page (REMOVE / 301 REDIRECT)": remove_url,
                    "Recommended Action": "Setup 301 Redirect to Primary Page and remove duplicate from sitemap"
                })

    # 6. Export to CSV (Bypasses Excel Limits) & Excel
    df = pd.DataFrame(report_data)

    if not df.empty:
        df.sort_values(by="Similarity Value", ascending=False, inplace=True)
        df.drop(columns=["Similarity Value"], inplace=True)
        
        # Save as CSV (Always safe) and Excel
        df.to_csv("sitemap_duplicates_report.csv", index=False)
        df.to_excel(OUTPUT_EXCEL, index=False)

        elapsed_min = round((time.time() - start_time) / 60, 2)
        print(f"\n✅ COMPLETED IN {elapsed_min} MINUTES!")
        print(f"📊 Report generated successfully with {len(df)} unique duplicate pages flagged.")
        print(f"   Saved to: '{OUTPUT_EXCEL}' and 'sitemap_duplicates_report.csv'")
    else:
        print("\n✅ COMPLETED! No duplicate content pairs met or exceeded the 90% threshold.")


if __name__ == "__main__":
    run_master_sitemap_analysis()