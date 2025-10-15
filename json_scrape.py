import requests
from bs4 import BeautifulSoup
import json
import time, random


BASE_URL = "https://laws.boe.gov.sa/"
BASE_EXTEN = "BoeLaws/Laws/Folders"

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
REQUEST_KWARGS = {'headers':HEADERS, 'timeout':20}

def scrape_individual_law_page(law_url):
    """Takes a URL to a specific law and its title, scrapes all its articles."""
    print(f"    Scraping articles from: {law_url}")
    try:
        # Using a common user-agent header
        
        response = requests.get(law_url, **REQUEST_KWARGS)
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all article containers
        articles = soup.find_all('div', class_='article_item')
        
        if not articles:
            print(f"     -> No articles found on page: {law_url}")
            return []

        scraped_articles = []
        for article_div in articles:
            title_tag = article_div.find('h3', class_='center')
            title = title_tag.get_text(strip=True) if title_tag else 'No Title'

            # Remove any popup divs that might interfere with text extraction
            popups = article_div.find_all('div', class_='popup-list')
            for popup in popups:
                popup.decompose()

            text_container = article_div.find('div', class_='HTMLContainer')
            if text_container:
                # Prepend numbers to list items for clarity
                for ol in text_container.find_all('ol'):
                    for i, li in enumerate(ol.find_all('li'), 1):
                        li.insert(0, f"{i}. ")
                text = text_container.get_text(separator='\n', strip=True)
            else:
                text = 'No Text Found'

            scraped_articles.append({
                'Article_Title': title,
                'Article_Text': text,
            })
        return scraped_articles
        
    except requests.exceptions.RequestException as e:
        print(f"     -> Error scraping {law_url}: {e}")
        return []

def scrape_boe_laws():
    """
    Parses the main laws page, extracts the hierarchy, and scrapes each individual law page.
    """
    
    laws_hierarchy = {}

    response = requests.get(BASE_URL + BASE_EXTEN, **REQUEST_KWARGS)
    html_content = response.text


    soup = BeautifulSoup(html_content, 'html.parser')

    # Main categories are in the vertical tab navigation
    main_categories_tags = soup.select("#vertical_tab_nav > ul > li > a")
    # Content for these categories is in the corresponding <article> tags
    articles_content = soup.select("#vertical_tab_nav > div.tab-content > article")

    # Map main category names to their content blocks
    for i, main_category_tag in enumerate(main_categories_tags):
        main_category_name = main_category_tag.get_text(strip=True)
        print(f"\nProcessing Main Category: {main_category_name}")
        
        laws_hierarchy[main_category_name] = {}
        
        if i < len(articles_content):
            article = articles_content[i]
            
            # Find all sub-categories within the current main category article
            sub_category_divs = article.find_all('div', class_=lambda x: x and x.startswith('content-'))
            
            for sub_div in sub_category_divs:
                # Get sub-category title
                link_div = sub_div.find('div', class_='link')
                if not link_div:
                    continue

                sub_category_name = link_div.get_text(strip=True)
                print(f"  -> Found Sub-Category: {sub_category_name}")
                laws_hierarchy[main_category_name][sub_category_name] = {}
                
                # Find all law links within this sub-category
                law_links = sub_div.select(".submenu ul li a")
                
                for link in law_links:
                    law_title = link.get_text(strip=True)
                    relative_url = link.get('href')
                    
                    if relative_url:
                        full_url = BASE_URL + relative_url
                        # Scrape the individual law page and add its content to the hierarchy
                        scraped_data = scrape_individual_law_page(full_url)
                        time.sleep(random.uniform(0,2))
                        laws_hierarchy[main_category_name][sub_category_name][law_title] = scraped_data

    # Save the complete hierarchy to a JSON file
    output_filename = "saudi_laws_scraped.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(laws_hierarchy, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Scraping complete. All data saved to '{output_filename}'")


if __name__ == "__main__":
    # Start the scraping process using the local HTML file
    scrape_boe_laws()
