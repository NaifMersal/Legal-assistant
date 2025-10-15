import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

# --- Configuration ---
BASE_URL = "https://laws.boe.gov.sa"
# URL for the "العمل والرعاية الاجتماعية" category
STARTING_INDEX_URL = "https://laws.boe.gov.sa/BoeLaws/Laws/Folders/1?lawClassificationId=3"
CSV_OUTPUT_FILE = 'all_labor_laws_consolidated.csv'

def scrape_individual_law_page(law_url, law_title):
    """Takes a URL to a specific law and its title, scrapes all its articles."""
    print(f"   scraping articles from: {law_url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(law_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('div', class_='article_item')
        
        scraped_articles = []
        for article_div in articles:
            title_tag = article_div.find('h3', class_='center')
            title = title_tag.get_text(strip=True) if title_tag else 'No Title'

            popups = article_div.find_all('div', class_='popup-list')
            for popup in popups:
                popup.decompose()

            text_container = article_div.find('div', class_='HTMLContainer')
            if text_container:
                for ol in text_container.find_all('ol'):
                    for i, li in enumerate(ol.find_all('li'), 1):
                        li.insert(0, f"{i}. ")
                text = text_container.get_text(separator='\n', strip=True)
            else:
                text = 'No Text Found'

            scraped_articles.append({
                'Source_Law': law_title,
                'Source_URL': law_url,
                'Article_Title': title,
                'Article_Text': text,
            })
        return scraped_articles
    except requests.exceptions.RequestException as e:
        print(f"    -> Error scraping {law_url}: {e}")
        return []

def main():
    """Main function to orchestrate the multi-page scraping process."""
    all_law_links = []
    current_page_url = STARTING_INDEX_URL
    page_num = 1

    print("--- Starting Phase 1: Discovering all law links ---")
    while current_page_url:
        print(f"-> Fetching law list from page {page_num}: {current_page_url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(current_page_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # --- CORRECTED LOGIC TO FIND LINKS ---
            # This selector specifically finds <a> tags whose href contains the path to law details pages.
            links_on_page = soup.select('a[href*="/BoeLaws/Laws/LawDetails/"]')
            
            for link in links_on_page:
                law_title = link.get_text(strip=True)
                relative_url = link.get('href')
                if relative_url:
                    full_url = urljoin(BASE_URL, relative_url)
                    # Avoid adding duplicate links if they appear multiple times
                    if not any(d['url'] == full_url for d in all_law_links):
                        all_law_links.append({'title': law_title, 'url': full_url})
            
            print(f"   Found {len(links_on_page)} laws on this page.")

            next_page_link = soup.find('a', string='التالي')
            if next_page_link and next_page_link.get('href'):
                current_page_url = urljoin(BASE_URL, next_page_link['href'])
                page_num += 1
                time.sleep(1) 
            else:
                current_page_url = None 
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching index page {current_page_url}: {e}")
            break

    print(f"\n--- Phase 1 Complete: Discovered a total of {len(all_law_links)} unique laws to scrape. ---\n")

    if not all_law_links:
        print("No law links were found. Exiting.")
        return

    print("--- Starting Phase 2: Scraping articles from each law ---")
    all_articles_data = []
    for i, law in enumerate(all_law_links, 1):
        print(f"Scraping law {i}/{len(all_law_links)}: '{law['title']}'")
        articles_from_law = scrape_individual_law_page(law['url'], law['title'])
        all_articles_data.extend(articles_from_law)
        time.sleep(1)

    print(f"\n--- Phase 2 Complete: Scraped a total of {len(all_articles_data)} articles. ---\n")

    if not all_articles_data:
        print("No articles were scraped. The CSV file will not be created.")
        return

    print(f"--- Starting Phase 3: Writing all data to {CSV_OUTPUT_FILE} ---")
    fieldnames = ['Source_Law', 'Source_URL', 'Article_Title', 'Article_Text']
    with open(CSV_OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_articles_data)
    
    print(f"✅ Success! All data has been saved to '{CSV_OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()