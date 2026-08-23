import json
import os
import re
import time
import gc
import urllib.parse
import urllib.request
import pandas as pd
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configuration Files & Directories
DUPLICATE_REPORT_CSV = "sitemap_duplicates_report.csv"
PLATFORM_REPORT_CSV = "platform_split_report.csv"
DB_DIR = "./chroma_db_bk"
CACHE_FILE = "cached_pages.json"

def clean_wordpress_shortcodes(text: str) -> str:
    """Removes WordPress shortcodes like [drts-directory-search ...] from text."""
    clean_text = re.sub(r'\[[a-zA-Z0-9_\-]+(?:\s+[^\]]+)?\]', '', text)
    return re.sub(r'\n\s*\n', '\n\n', clean_text).strip()

def direct_google_translate(text: str) -> str:
    """Translates text via direct HTTP request to avoid external package locks."""
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + urllib.parse.quote(text)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            return "".join([sentence[0] for sentence in result[0] if sentence[0]])
    except Exception:
        return text

def translate_if_hindi(text: str) -> str:
    """Detects Devanagari Hindi text and safely translates it in chunks."""
    if any('\u0900' <= char <= '\u097F' for char in text):
        time.sleep(0.3)  # Gentle delay to prevent IP block
        if len(text) > 1500:
            blocks = [text[i:i+1500] for i in range(0, len(text), 1500)]
            translated_blocks = []
            for b in blocks:
                translated_blocks.append(direct_google_translate(b))
                time.sleep(0.2)
            return " ".join(translated_blocks)
        else:
            return direct_google_translate(text)
    return text

def get_clean_primary_urls():
    """Reads sitemap CSVs, excludes duplicates, and explicitly appends the main BK One portal."""
    urls = set()
    
    if os.path.exists(PLATFORM_REPORT_CSV):
        df_platform = pd.read_csv(PLATFORM_REPORT_CSV)
        if "URL" in df_platform.columns:
            urls.update(df_platform["URL"].dropna().tolist())
            print(f"-> Loaded {len(urls)} URLs from '{PLATFORM_REPORT_CSV}'.")

    if os.path.exists(DUPLICATE_REPORT_CSV):
        df_dup = pd.read_csv(DUPLICATE_REPORT_CSV)
        target_col = "Duplicate Page (REMOVE / 301 REDIRECT)"
        if target_col in df_dup.columns:
            remove_urls = set(df_dup[target_col].dropna().tolist())
            before_count = len(urls)
            urls = urls - remove_urls
            print(f"-> Excluded {before_count - len(urls)} duplicate URLs using '{DUPLICATE_REPORT_CSV}'.")

    # Explicitly include the BK One main landing portal
    bkone_url = "https://www.brahmakumaris.com/bkone"
    urls.add(bkone_url)
    print(f"-> Explicitly added landing page: {bkone_url}")

    clean_urls = list(urls)
    print(f"-> Total Clean Primary URLs to Index: {len(clean_urls)}")
    return clean_urls

def scrape_pages_with_cache(urls):
    """Loads cached text content if available; scrapes missing URLs via Playwright if available."""
    pages_data = {}
    if os.path.exists(CACHE_FILE):
        print(f"-> Reading cached content from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            pages_data = json.load(f)

    urls_to_scrape = [u for u in urls if u not in pages_data]
    
    if urls_to_scrape:
        print(f"-> Found {len(urls_to_scrape)} missing URLs to scrape.")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                for idx, url in enumerate(urls_to_scrape, 1):
                    print(f"   [{idx}/{len(urls_to_scrape)}] Fetching: {url}")
                    try:
                        page.goto(url, wait_until="networkidle", timeout=30000)
                        soup = BeautifulSoup(page.content(), 'html.parser')

                        for a in soup.find_all('a', href=True):
                            link_text = a.get_text(strip=True)
                            link_url = urljoin(url, a['href'])
                            if link_text and not a['href'].startswith("#"):
                                a.replace_with(f" [{link_text}]({link_url}) ")

                        for elem in soup(["script", "style", "nav", "footer", "header", "svg"]):
                            elem.extract()

                        clean_text = "\n".join([line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()])
                        clean_text = clean_wordpress_shortcodes(clean_text)

                        if len(clean_text) > 150:
                            pages_data[url] = clean_text
                    except Exception as e:
                        print(f"   ⚠️ Skipped {url}: {e}")

                browser.close()
        except Exception as e:
            print(f"⚠️ Playwright browser launch skipped in this environment: {e}")

    documents = []
    print("-> Processing content into English vector chunks...")
    cache_updated = False

    for idx, url in enumerate(urls, 1):
        if url in pages_data:
            content = clean_wordpress_shortcodes(pages_data[url])
            english_content = translate_if_hindi(content)
            
            if english_content != content:
                pages_data[url] = english_content
                cache_updated = True
            
            url_keywords = url.split('/')[-2].replace('-', ' ') if '/' in url else ""
            formatted_content = f"Page Source Link: {url}\nURL Topic Terms: {url_keywords}\n\n{english_content}"
            documents.append(Document(page_content=formatted_content, metadata={"source": url}))

    return documents

def remove_directory_safely(dir_path):
    """Safely removes Chroma database directory by freeing locks."""
    if os.path.exists(dir_path):
        gc.collect()
        time.sleep(1)
        
        for attempt in range(3):
            try:
                import shutil
                shutil.rmtree(dir_path)
                print(f"-> Cleared previous database at '{dir_path}'.")
                break
            except Exception:
                time.sleep(1)

def build_full_clean_vector_db():
    target_urls = get_clean_primary_urls()
    if not target_urls:
        print("❌ No valid URLs found.")
        return

    documents = scrape_pages_with_cache(target_urls)
    print(f"\n1. Loaded {len(documents)} clean primary page documents.")

    print("2. Chunking text content...")
    # Optimized chunk size to keep SQLite file well below GitHub's 100MB limit
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"   Generated {len(chunks)} searchable chunks.")

    print("3. Generating embeddings & saving ChromaDB locally...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    remove_directory_safely(DB_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"\n✅ SUCCESS: Full Clean Vector Database built at '{DB_DIR}'!")

if __name__ == "__main__":
    build_full_clean_vector_db()