import requests
from bs4 import BeautifulSoup
import json
import time, random


BASE_URL = "https://laws.boe.gov.sa/"
BASE_EXTEN = "BoeLaws/Laws/Folders"

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
REQUEST_KWARGS = {'headers':HEADERS, 'timeout':20, 'verify':False}



def scrape_individual_law_page(law_url):
    """
    Takes a URL to a specific law, scrapes its metadata and all its articles,
    preserving the Part/Chapter hierarchy and grouping articles by Part.
    """
    print(f"    Scraping articles from: {law_url}")
    try:
        response = requests.get(law_url, **REQUEST_KWARGS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # --- 1. Scrape Metadata ---

        system_details = soup.find('div', class_='system_details_box')
        if system_details:
            brief_tag = system_details.find('div', class_='system_brief')
            if brief_tag:
                brief_text_tag = brief_tag.find('div', class_='HTMLContainer')
                brief = brief_text_tag.get_text(separator='\n', strip=True) if brief_text_tag else ''

            info_div = system_details.find('div', class_='system_info')
            info_items = {}
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
                                    'url': (BASE_URL.rstrip('/') + a['href'])
                                })
                        info_items[current_key] = tools
                            
        
        # --- End Metadata ---

        # --- 2. Scrape Articles and Group by Part ---
        law_content_container = soup.find('div', id='divLawText')
        if not law_content_container:
            print(f"    -> No 'divLawText' container found. Page structure may have changed.")
            return {'brief':brief, 'metadata': info_items, 'parts': {}}

        law_content = law_content_container.find('div', recursive=False)
        if law_content is None:
            law_content = law_content_container

        parts_dict = {}
        # Default key for decrees, preambles, or laws without "Part" divisions
        current_part_title = 'Preamble' 

        for element in law_content.children:
            if element.name is None:
                continue

            # --- START FIX ---
            # Check if this element is a heading
            if element.name in ['h1', 'h3'] and 'center' in element.get('class', []):
                heading_text = element.get_text(strip=True)
                

                current_part_title = heading_text
                
                # This element is a heading (either a Part or just a title), 
                # not an article. Skip to the next element.
                continue
            # --- END FIX ---

            # This element is an article container
            if element.name == 'div' and 'article_item' in element.get('class', []):
                article_div = element
                
                all_titles = article_div.find_all('h3', class_='center')
                
                article_title = 'No Title'


                if all_titles:
                    article_title = all_titles[-1].get_text(strip=True)


                for popup in article_div.find_all('div', class_='popup-list'):
                    popup.decompose()

                text_container = article_div.find('div', class_='HTMLContainer')
                if text_container:
                    for ol in text_container.find_all('ol'):
                        for i, li in enumerate(ol.find_all('li'), 1):
                            if not li.get_text(strip=True).startswith(f"{i}."):
                                li.insert(0, f"{i}. ")
                    text = text_container.get_text(separator='\n', strip=True)
                else:
                    text = 'No Text Found'

                # 1. Check if this part is in our dictionary yet. If not, create it.
                if current_part_title not in parts_dict:
                    parts_dict[current_part_title] = []

                # 2. Append the article data to the list for the current part.
                parts_dict[current_part_title].append({
                    'Article_Title': article_title,
                    'Article_Text': text,
                })

        if not parts_dict:
            print(f"    -> No article items found inside 'divLawText' for: {law_url}")

        # Return the new structure
        return {'brief':brief, 'metadata': info_items, 'parts': parts_dict}

    except requests.exceptions.RequestException as e:
        print(f"    -> Error scraping {law_url}: {e}")
        return {'brief':'','metadata': {}, 'parts': {}}


def scrape_boe_laws():
    """
    Parses the main laws page, extracts the hierarchy, and scrapes each individual law page.
    """
    
    laws_hierarchy = {}

    print(f"Fetching main law index from: {BASE_URL + BASE_EXTEN}")
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
                        full_url = BASE_URL.rstrip('/') + relative_url
                        # Scrape the individual law page and add its content to the hierarchy
                        # The scraped_data will now be the full dict: {'metadata': {...}, 'articles': [...]}
                        scraped_data = scrape_individual_law_page(full_url)
                        time.sleep(random.uniform(0,2))
                        laws_hierarchy[main_category_name][sub_category_name][law_title] = scraped_data

    # Save the complete hierarchy to a JSON file
    output_filename = "saudi_laws_scraped.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(laws_hierarchy, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Scraping complete. All data saved to '{output_filename}'")
