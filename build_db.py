import json
import os
import re
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
    # Matches patterns like [shortcode_name key="value"] or [shortcode]
    clean_text = re.sub(r'\[[a-zA-Z0-9_\-]+(?:\s+[^\]]+)?\]', '', text)
    # Remove leftover multiple spaces or empty lines
    return re.sub(r'\n\s*\n', '\n\n', clean_text).strip()

def get_clean_primary_urls():
    """Reads sitemap CSVs and excludes duplicate/redirect URLs."""
    urls = set()
    
    if os.path.exists(PLATFORM_REPORT_CSV):
        df_platform = pd.read_csv(PLATFORM_REPORT_CSV)
        if "URL" in df_platform.columns:
            urls.update(df_platform["URL"].dropna().tolist())
            print(f"-> Loaded {len(urls)} URLs from '{PLATFORM_REPORT_CSV}'.")
    else:
        print(f"⚠️ Warning: '{PLATFORM_REPORT_CSV}' not found.")

    if os.path.exists(DUPLICATE_REPORT_CSV):
        df_dup = pd.read_csv(DUPLICATE_REPORT_CSV)
        target_col = "Duplicate Page (REMOVE / 301 REDIRECT)"
        if target_col in df_dup.columns:
            remove_urls = set(df_dup[target_col].dropna().tolist())
            before_count = len(urls)
            urls = urls - remove_urls
            print(f"-> Excluded {before_count - len(urls)} duplicate URLs using '{DUPLICATE_REPORT_CSV}'.")

    clean_urls = list(urls)
    print(f"-> Total Clean Primary URLs to Index: {len(clean_urls)}")
    return clean_urls

def scrape_pages_with_cache(urls):
    """Loads cached text content if available; scrapes missing URLs via Playwright."""
    pages_data = {}
    if os.path.exists(CACHE_FILE):
        print(f"-> Reading cached content from '{CACHE_FILE}'...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            pages_data = json.load(f)

    urls_to_scrape = [u for u in urls if u not in pages_data]
    
    if urls_to_scrape:
        print(f"-> Scraping {len(urls_to_scrape)} new/uncached pages with Playwright...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            for idx, url in enumerate(urls_to_scrape, 1):
                print(f"   [{idx}/{len(urls_to_scrape)}] Fetching: {url}")
                try:
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    soup = BeautifulSoup(page.content(), 'html.parser')

                    # Preserve links in Markdown format [Text](URL)
                    for a in soup.find_all('a', href=True):
                        link_text = a.get_text(strip=True)
                        link_url = urljoin(url, a['href'])
                        if link_text and not a['href'].startswith("#"):
                            a.replace_with(f" [{link_text}]({link_url}) ")

                    # Strip layout noise
                    for elem in soup(["script", "style", "nav", "footer", "header", "svg"]):
                        elem.extract()

                    clean_text = "\n".join([line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()])
                    
                    # Clean out WordPress shortcodes immediately after extraction
                    clean_text = clean_wordpress_shortcodes(clean_text)

                    if len(clean_text) > 150:
                        pages_data[url] = clean_text
                except Exception as e:
                    print(f"   ⚠️ Skipped {url}: {e}")

            browser.close()

        # Update cache on disk
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=2)

    # Convert dictionary into LangChain Document instances with explicitly attached URLs
    documents = []
    for url in urls:
        if url in pages_data:
            # Clean cached content to purge any shortcodes if cache was built earlier
            content = clean_wordpress_shortcodes(pages_data[url])
            
            # Prepend exact source URL directly to the document text
            formatted_content = f"Page Source Link: {url}\n\n{content}"
            documents.append(Document(page_content=formatted_content, metadata={"source": url}))
            
    return documents

def build_full_clean_vector_db():
    target_urls = get_clean_primary_urls()
    if not target_urls:
        print("❌ No valid URLs found. Make sure your report CSV files are in the folder.")
        return

    documents = scrape_pages_with_cache(target_urls)
    print(f"\n1. Loaded {len(documents)} clean primary page documents.")

    print("2. Chunking text content...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"   Generated {len(chunks)} searchable chunks.")

    print("3. Generating embeddings & saving ChromaDB locally...")
    # Updated to Multilingual Model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if os.path.exists(DB_DIR):
        try:
            import shutil
            shutil.rmtree(DB_DIR)
        except Exception as e:
            print(f"⚠️ Could not automatically delete '{DB_DIR}': {e}")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"\n✅ SUCCESS: Full Clean Vector Database built at '{DB_DIR}'!")

if __name__ == "__main__":
    build_full_clean_vector_db()