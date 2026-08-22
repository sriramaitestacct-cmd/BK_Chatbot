import concurrent.futures
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

MASTER_SITEMAP = "https://www.brahmakumaris.com/sitemap_index.xml"
MAX_WORKERS = 10
OUTPUT_CSV = "platform_split_report.csv"


def get_all_urls(master_url):
    """Fetches all URLs from the master sitemap index."""
    print(f"-> Accessing Master Sitemap: {master_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
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

        for sitemap in child_sitemaps:
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
            except Exception:
                pass
    except Exception as e:
        print(f"❌ Error fetching sitemaps: {e}")

    unique_urls = list(set(all_urls))
    print(f"Discovered {len(unique_urls)} total URLs across all sitemaps.\n")
    return unique_urls


def detect_tech_stack(url):
    """Detects if a URL is running on WordPress or Next.js."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        html = res.text.lower()
        response_headers = {k.lower(): v.lower() for k, v in res.headers.items()}

        # Next.js indicators
        is_next = (
            "__next_data__" in html
            or "/_next/static/" in html
            or "next.js" in response_headers.get("x-powered-by", "")
        )

        # WordPress indicators
        is_wp = (
            "wp-content" in html
            or "wp-includes" in html
            or "wordpress" in html
        )

        if is_next:
            platform = "Next.js (Modern)"
        elif is_wp:
            platform = "WordPress (Legacy)"
        else:
            platform = "Other / Static HTML"

        return {"URL": url, "Platform": platform, "HTTP Status": res.status_code}
    except Exception:
        return {"URL": url, "Platform": "Unreachable / Error", "HTTP Status": "Failed"}


def run_platform_audit():
    urls = get_all_urls(MASTER_SITEMAP)
    if not urls:
        return

    print(f"Analyzing platform tech stack for {len(urls)} pages...")
    results = []
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(detect_tech_stack, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            results.append(future.result())
            completed += 1
            if completed % 200 == 0 or completed == len(urls):
                print(f"   Progress: [{completed}/{len(urls)}] pages checked...")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    # Calculate and Display Summary Split
    print("\n" + "=" * 40)
    print("      PLATFORM SPLIT SUMMARY")
    print("=" * 40)
    summary = df["Platform"].value_counts()
    for platform, count in summary.items():
        percentage = round((count / len(urls)) * 100, 2)
        print(f"• {platform}: {count} pages ({percentage}%)")
    print("=" * 40)
    print(f"\n✅ Detailed URL report saved to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    run_platform_audit()