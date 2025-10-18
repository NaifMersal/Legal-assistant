import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import json
import time
import random
import logging
from urllib.parse import urljoin
import urllib3

# --- Configuration ---

# Disable the warning for 'verify=False'. 
# Note: It's better to remove 'verify=False' entirely if possible.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://laws.boe.gov.sa/"
LAWS_PATH = "BoeLaws/Laws/Folders"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraping.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- Helper Function: Individual Law Page ---
def scrape_individual_law_page(session, law_url, start_id):
    """
    Scrapes metadata and articles for a specific law URL using a provided session.
    
    Includes fallback logic for pages without 'article_item' divs.
    
    Args:
        session (requests.Session): The session object to use for the request.
        law_url (str): The URL of the law page to scrape.
        start_id (int): The starting ID for articles.
        
    Returns:
        tuple: A tuple containing (scraped_data, last_id).
               scraped_data is a dict with {'brief', 'metadata', 'parts'}.
               last_id is the new ID counter after processing this page.
    """
    logging.info(f"Scraping articles from: {law_url}")
    id_counter = start_id
    empty_data = {'brief': '', 'metadata': {}, 'parts': {}}
    
    try:
        response = session.get(law_url)
        response.raise_for_status()
    except RequestException as e:
        logging.error(f"Failed to fetch {law_url}: {e}")
        return empty_data, start_id

    try:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # --- 1. Scrape Metadata (Your original code is fine) ---
        brief = ''
        info_items = {}
        system_details = soup.find('div', class_='system_details_box')
        
        if system_details:
            brief_tag = system_details.find('div', class_='system_brief')
            if brief_tag:
                brief_text_tag = brief_tag.find('div', class_='HTMLContainer')
                if brief_text_tag:
                    brief = brief_text_tag.get_text(separator='\n', strip=True)

            info_div = system_details.find('div', class_='system_info')
            if info_div:
                for div in info_div.find_all('div', recursive=False):
                    label_tag = div.find('label')
                    if not label_tag:
                        continue
                    
                    current_key = label_tag.get_text(strip=True)
                    span_tag = div.find('span')
                    if span_tag:
                        info_items[current_key] = span_tag.get_text(strip=True)
                    
                    ul = div.find('ul')
                    if ul:
                        tools = []
                        for li in ul.find_all('li'):
                            a = li.find('a')
                            if a and a.get('href'):
                                tools.append({
                                    'text': a.get_text(strip=True),
                                    'url': urljoin(BASE_URL, a['href'])
                                })
                        info_items[current_key] = tools

        # --- 2. Scrape Articles ---
        parts_dict = {}
        current_part_title = 'main' # Default key
        law_content_container = soup.find('div', id='divLawText')
        
        if not law_content_container:
            logging.warning(f"No 'divLawText' container found at {law_url}")
            return {'brief': brief, 'metadata': info_items, 'parts': {}}, id_counter

        law_content = law_content_container.find('div', recursive=False)
        if law_content is None:
            law_content = law_content_container

        # --- Primary Scrape Logic (Your original code) ---
        for element in law_content.children:
            if element.name is None:
                continue

            if element.name in ['h1', 'h3'] and 'center' in element.get('class', []):
                current_part_title = element.get_text(strip=True)
                continue 

            if element.name == 'div' and 'article_item' in element.get('class', []):
                all_titles = element.find_all('h3', class_='center')
                article_title = all_titles[-1].get_text(strip=True) if all_titles else 'No Title'

                element_classes = element.get('class', [])
                status = "Active"  
                if "canceled-article" in element_classes:
                    status = "Canceled"
                elif "changed-article" in element_classes:
                    status = "Modified"
                
                for popup in element.find_all('div', class_='popup-list'):
                    popup.decompose()

                text_container = element.find('div', class_='HTMLContainer')
                if text_container:
                    for ol in text_container.find_all('ol'):
                        for i, li in enumerate(ol.find_all('li'), 1):
                            if not li.get_text(strip=True).startswith(f"{i}."):
                                li.insert(0, f"{i}. ")
                    text = text_container.get_text(separator='\n', strip=True)
                else:
                    text = 'No Text Found'

                if current_part_title not in parts_dict:
                    parts_dict[current_part_title] = []

                parts_dict[current_part_title].append({
                    'id': id_counter,
                    'Article_Title': article_title.replace(':', '').strip(),
                    'status': status,
                    'Article_Text': '\n'.join(line.strip() for line in text.splitlines() if line.strip())
                })
                id_counter += 1

        # --- FALLBACK LOGIC START (NEW) ---
        # If the loop above found no 'article_item' divs, parts_dict will be empty.
        if not parts_dict:
            logging.warning(f"No 'article_item' divs found at {law_url}. Switching to fallback mode.")
            
            if law_content:
                # Clean up popups, just in case
                for popup in law_content.find_all('div', class_='popup-list'):
                    popup.decompose()
                
                # Get the *entire* text of the content block
                full_text = law_content.get_text(separator='\n', strip=True)
                cleaned_text = '\n'.join(line.strip() for line in full_text.splitlines() if line.strip())
                
                if cleaned_text:
                    # Add this full text as a single "article"
                    parts_dict[current_part_title] = [{
                        'id': id_counter,
                        'Article_Title': 'نص اللائحة', # "Text of the Regulation"
                        'status': info_items.get('الحالة', 'Active'), # Assume active
                        'Article_Text': cleaned_text
                    }]
                    id_counter += 1 # Increment the counter once for this "article"
                else:
                    logging.error(f"Fallback failed: 'divLawText' was empty for {law_url}")
        # --- FALLBACK LOGIC END ---

        # --- Safely prepare data for return (MODIFIED) ---
        final_parts_dict = {}
        if not parts_dict:
            logging.error(f"Completely failed to extract any articles from {law_url}")
            # Return the original start_id since no articles were added
            return {'brief': brief, 'metadata': info_items, 'parts': {}}, start_id
        
        parts_keys = list(parts_dict.keys())
        
        # This logic now safely handles an empty or populated parts_dict
        if len(parts_keys) > 1:
            final_parts_dict = parts_dict
        else:
            # If 0 or 1 part, default to this structure
            # This fixes your original IndexError
            final_parts_dict = {'main': parts_dict[parts_keys[0]]}

        scraped_data = {'brief': brief, 'metadata': info_items, 'parts': final_parts_dict}
        return scraped_data, id_counter

    except Exception as e:
        logging.error(f"Error parsing page {law_url}: {e}", exc_info=True)
        return empty_data, start_id

# --- Main Scraping Function ---

def scrape_boe_laws():
    """
    Parses the main laws page, extracts the hierarchy, and scrapes each law.
    Saves the result to a JSON file.
    """
    laws_hierarchy = {}
    id_counter = 0 # Initialize ID counter here
    
    # Use a Session object
    with requests.Session() as session:
        # Set session properties
        session.headers.update(HEADERS)
        session.timeout = 20
        session.verify = False # Ideally, set this to True or remove it
        
        main_page_url = urljoin(BASE_URL, LAWS_PATH)
        logging.info(f"Fetching main law index from: {main_page_url}")
        
        try:
            response = session.get(main_page_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
        except RequestException as e:
            logging.critical(f"Failed to fetch main index page. Aborting. Error: {e}")
            return # Exit if we can't get the main page

        main_categories_tags = soup.select("#vertical_tab_nav > ul > li > a")
        articles_content = soup.select("#vertical_tab_nav > div.tab-content > article")

        for i, main_category_tag in enumerate(main_categories_tags):
            main_category_name = main_category_tag.get_text(strip=True)
            logging.info(f"\nProcessing Main Category: {main_category_name}")
            laws_hierarchy[main_category_name] = {}
            
            if i >= len(articles_content):
                logging.warning(f"No content article found for category: {main_category_name}")
                continue
                
            article = articles_content[i]
            sub_category_divs = article.find_all('div', class_=lambda x: x and x.startswith('content-'))
            
            for sub_div in sub_category_divs:
                link_div = sub_div.find('div', class_='link')
                if not link_div:
                    continue

                sub_category_name = link_div.get_text(strip=True)
                logging.info(f"  -> Found Sub-Category: {sub_category_name}")
                laws_hierarchy[main_category_name][sub_category_name] = {}
                
                law_links = sub_div.select(".submenu ul li a")
                
                for link in law_links:
                    law_title = link.get_text(strip=True)
                    relative_url = link.get('href')
                    
                    if relative_url:
                        full_url = urljoin(BASE_URL, relative_url)
                        
                        # Pass session and counter, get data and new counter back
                        scraped_data, id_counter = scrape_individual_law_page(session, full_url, id_counter)
                        
                        laws_hierarchy[main_category_name][sub_category_name][law_title] = scraped_data
                        
                        # Polite sleep
                        time.sleep(random.uniform(0.5, 1.5))

    # --- Save Output ---
    output_filename = "saudi_laws_scraped.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(laws_hierarchy, f, ensure_ascii=False, indent=4)
        logging.info(f"\n✅ Scraping complete. All data saved to '{output_filename}'")
    except IOError as e:
        logging.critical(f"Failed to write JSON to {output_filename}: {e}")



if __name__ == "__main__":
    scrape_boe_laws()