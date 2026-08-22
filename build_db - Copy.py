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
# ENV OPTIONS: "DEPTH" (for local testing) or "SITEMAP" (for production deployment)
CRAWL_MODE = os.getenv("CRAWL_MODE", "DEPTH")  

# Active only when CRAWL_MODE = "DEPTH"
# 0 = Homepage only
# 1 = Homepage + Direct sub-pages (e.g., /about/current-leaders)
# 2 = Sub-pages of sub-pages
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "1"))

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
                        
                        # Restrict crawl to same domain and exclude non-HTML media files
                        if urlparse(full_url).netloc == urlparse(start_url).netloc:
                            if not full_url.endswith(('.pdf', '.jpg', '.png', '.jpeg', '.mp4', '.webp')):
                                if full_url not in visited:
                                    to_visit.add((full_url, depth + 1))
                except Exception as e:
                    print(f"   [Error] Fetching links from {current_url}: {e}")

        browser.close()

    return list(visited)


def get_urls_from_sitemap(sitemap_url):
    """Extracts all published canonical URLs from official website sitemap."""
    print(f"-> Running PRODUCTION SITEMAP INGESTION ({sitemap_url})")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(sitemap_url, headers=headers)
        soup = BeautifulSoup(response.content, 'xml')
        
        # Extract loc nodes and filter out non-HTML files
        urls = [
            loc.text.strip() for loc in soup.find_all('loc') 
            if not loc.text.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.webp', '.mp4'))
        ]
        return list(set(urls))
    except Exception as e:
        print(f"   [Error] Failed to read sitemap: {e}")
        return [START_URL]


def fetch_page_content(page, url):
    """Scrapes dynamic JS rendered content, converts links, and strips noise."""
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

        # Strip layout/navigation elements
        for elem in soup(["script", "style", "nav", "footer", "header", "svg"]):
            elem.extract()

        clean_text = "\n".join([line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()])
        
        if clean_text:
            return Document(page_content=clean_text, metadata={"source": url, "title": page.title()})
    except Exception as e:
        print(f"   [Skipped] Error reading {url}: {e}")
        return None


def build_vector_db():
    # 1. Gather Target URLs based on active Mode
    if CRAWL_MODE == "SITEMAP":
        target_urls = get_urls_from_sitemap(SITEMAP_URL)
    else:
        target_urls = get_urls_by_depth(START_URL, MAX_DEPTH)

    print(f"\n1. Discovered {len(target_urls)} unique pages to index.")

    # 2. Extract content using a single Playwright Browser instance
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

    # 3. Chunk Documents
    print("\n2. Splitting content into searchable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_documents)

    # 4. Generate Embeddings & Store in ChromaDB
    print("3. Generating embeddings & writing to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_bk"
    )
    print("SUCCESS: Local Vector Database populated successfully!")


if __name__ == "__main__":
    build_vector_db()