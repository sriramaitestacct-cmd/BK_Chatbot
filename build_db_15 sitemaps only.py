import os
import time
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ==============================================================================
# CONFIGURATION SETTINGS
# ==============================================================================
# Set to "SITEMAP" to test sitemap ingestion or "DEPTH" for link depth crawling
CRAWL_MODE = os.getenv("CRAWL_MODE", "SITEMAP")  

# Active only when CRAWL_MODE = "DEPTH"
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "1"))

# DEV TESTING LIMITS FOR SITEMAPS (Set both to None for Production full indexing)
DEV_MAX_SITEMAPS = 2  # Inspect only the first 2 child sitemaps (site1.xml, site2.xml)
DEV_MAX_URLS = 15     # Limit to 15 total URLs across child sitemaps for fast dev testing

START_URL = "https://www.brahmakumaris.com"
SITEMAP_URL = "https://www.brahmakumaris.com/sitemap.xml"
# ==============================================================================


def get_urls_by_depth(start_url, max_depth):
    """Recursively collects internal URLs up to max_depth."""
    print(f"-> Running DEPTH CRAWLER (Target Max Depth: {max_depth})")
    visited = set()
    to_visit = {(start_url, 0)}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        while to_visit:
            current_url, depth = to_visit.pop()
            
            if current_url in visited or depth > max_depth:
                continue

            visited.add(current_url)
            print(f"   [Depth {depth}] Found URL: {current_url}")

            if depth < max_depth:
                try:
                    page.goto(current_url, wait_until="networkidle", timeout=30000)
                    soup = BeautifulSoup(page.content(), 'html.parser')

                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        full_url = urljoin(current_url, href).split('#')[0].rstrip('/')
                        
                        if urlparse(full_url).netloc == urlparse(start_url).netloc:
                            if not full_url.endswith(('.pdf', '.jpg', '.png', '.jpeg', '.mp4', '.webp')):
                                if full_url not in visited:
                                    to_visit.add((full_url, depth + 1))
                except Exception as e:
                    print(f"   [Error] Fetching links from {current_url}: {e}")

        browser.close()

    return list(visited)


def get_urls_from_sitemap(sitemap_url, max_sitemaps=None, max_urls=None):
    """
    Extracts page URLs from a master sitemap index or child sitemaps.
    Handles nested sitemap structures and applies sampling caps for dev environments.
    """
    print(f"-> Fetching Sitemap Index: {sitemap_url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    collected_urls = set()

    try:
        res = requests.get(sitemap_url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, 'xml')

        # Find child sitemaps inside <sitemap><loc>...</loc></sitemap>
        sitemap_tags = soup.find_all('sitemap')
        
        if sitemap_tags:
            sub_sitemaps = [s.find('loc').text.strip() for s in sitemap_tags if s.find('loc')]
            print(f"   Found {len(sub_sitemaps)} child sitemaps in master index.")

            if max_sitemaps and len(sub_sitemaps) > max_sitemaps:
                print(f"   [DEV MODE] Inspecting only the first {max_sitemaps} child sitemaps.")
                sub_sitemaps = sub_sitemaps[:max_sitemaps]

            for sitemap in sub_sitemaps:
                print(f"   Reading child sitemap: {sitemap}")
                child_res = requests.get(sitemap, headers=headers, timeout=15)
                child_soup = BeautifulSoup(child_res.content, 'xml')

                # Extract individual page URLs from <url><loc>...</loc></url>
                urls = [
                    loc.text.strip() for loc in child_soup.find_all('loc')
                    if loc.text and not loc.text.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.webp', '.mp4'))
                ]
                collected_urls.update(urls)

                if max_urls and len(collected_urls) >= max_urls:
                    print(f"   [DEV MODE] Reached URL threshold ({max_urls} page URLs). Stopping sitemap crawl.")
                    break
        else:
            # Single flat sitemap fallback
            urls = [
                loc.text.strip() for loc in soup.find_all('loc')
                if loc.text and not loc.text.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.webp', '.mp4'))
            ]
            collected_urls.update(urls)

    except Exception as e:
        print(f"   [Error] Failed to read sitemap: {e}")
        return []

    final_urls = list(collected_urls)

    if max_urls and len(final_urls) > max_urls:
        final_urls = final_urls[:max_urls]

    print(f"-> Total unique sitemap URLs collected: {len(final_urls)}")
    return final_urls


def fetch_page_content(page, url):
    """Scrapes dynamic JS-rendered content, converts <a> tags to markdown, and strips noise."""
    try:
        page.goto(url, wait_until="networkidle", timeout=45000)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
        time.sleep(1.5)

        soup = BeautifulSoup(page.content(), 'html.parser')

        # Convert anchor tags to Markdown format [Text](URL)
        for a in soup.find_all('a', href=True):
            link_text = a.get_text(strip=True)
            link_url = urljoin(url, a['href'])
            if link_text and not a['href'].startswith("#"):
                a.replace_with(f" [{link_text}]({link_url}) ")

        # Remove layout/navigation elements
        for elem in soup(["script", "style", "nav", "footer", "header", "svg"]):
            elem.extract()

        clean_text = "\n".join([line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()])
        
        if clean_text:
            return Document(page_content=clean_text, metadata={"source": url, "title": page.title()})
    except Exception as e:
        print(f"   [Skipped] Error reading {url}: {e}")
        return None


def build_vector_db():
    if CRAWL_MODE == "SITEMAP":
        sitemap_urls = get_urls_from_sitemap(
            SITEMAP_URL, 
            max_sitemaps=DEV_MAX_SITEMAPS, 
            max_urls=DEV_MAX_URLS
        )
        # Combines Homepage (START_URL) + 15 sitemap URLs while eliminating duplicates
        target_urls = list(set([START_URL] + sitemap_urls))
    else:
        target_urls = get_urls_by_depth(START_URL, MAX_DEPTH)

    print(f"\n1. Discovered {len(target_urls)} unique pages to index (including Homepage).")

    raw_documents = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for i, url in enumerate(target_urls, 1):
            print(f"[{i}/{len(target_urls)}] Scraping: {url}")
            doc = fetch_page_content(page, url)
            if doc:
                raw_documents.append(doc)
        browser.close()

    if not raw_documents:
        print("Scraping failed: No valid page contents retrieved.")
        return

    print("\n2. Splitting content into searchable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_documents)

    print("3. Generating embeddings & writing to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_bk"
    )
    print("\nSUCCESS: Local Vector Database populated successfully!")


if __name__ == "__main__":
    build_vector_db()