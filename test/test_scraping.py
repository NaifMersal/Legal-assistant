import csv
import requests
from bs4 import BeautifulSoup

def scrape_law_articles(html_content):
    """
    Parses an HTML file to extract law articles and saves them to a CSV file.

    Args:
        html_file_path (str): The path to the input HTML file.
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all the article containers
    articles = soup.find_all('div', class_='article_item')

    # Prepare a list to hold the extracted article data
    all_articles_data = []

    # Loop through each article found
    for article_div in articles:
        # Extract the article title
        title_tag = article_div.find('h3', class_='center')
        title = title_tag.get_text(strip=True) if title_tag else 'No Title'

        # 
        # --- THIS IS THE CORRECTED LINE ---
        # Find the HTMLContainer that is a DIRECT CHILD of article_div, ignoring nested ones.
        #
        text_container = article_div.find('div', class_='HTMLContainer', recursive=False)
        
        text = text_container.get_text(separator='\n', strip=True) if text_container else 'No Text Found'

        # Store the extracted title and final text
        all_articles_data.append({
            'Article_Title': title,
            'Article_Text': text,
        })

    # Define the CSV file name and the headers
    csv_file_name = 'law_articles_corrected.csv'
    fieldnames = ['Article_Title', 'Article_Text']

    # Write the data to the CSV file
    try:
        with open(csv_file_name, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_articles_data)
            
        print(f"✅ Successfully scraped {len(all_articles_data)} articles and saved the corrected versions to '{csv_file_name}'")

    except IOError:
        print(f"Error: Could not write to the file '{csv_file_name}'.")
















# --- How to use the script ---
if __name__ == "__main__":
    # 1. Save the provided HTML content into a file named 'law_page.html'
    # 2. Place this Python script in the same directory.
    # 3. Run the script. It will create 'law_articles.csv'.

    HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BOE-ArticleScraper/1.0)"
}
    response = requests.get("https://laws.boe.gov.sa/BoeLaws/Laws/LawDetails/08381293-6388-48e2-8ad2-a9a700f2aa94/1", headers=HEADERS)
    input_html = response.text

    scrape_law_articles(input_html)















    # edited old code - IGNORE::::::::::::



        # try:
    #     soup = BeautifulSoup(html_content, 'html.parser')

    #     # Step 3: Find all article containers
    #     article_divs = soup.find_all('div', class_='article_item')

    #     articles_data = []
    #     print(f"Found {len(article_divs)} article containers to process.")

    #     # Step 4: Loop through each article and extract only the primary text
    #     for article in article_divs:
    #         title_tag = article.find('h3', class_='center')
            
    #         # Use .find() to get the FIRST 'HTMLContainer', which is the main text box.
    #         # This intentionally skips the amendment text in the nested popup.
    #         text_tag = article.find('div', class_='HTMLContainer')

    #         if title_tag and text_tag:
    #             title = title_tag.get_text(strip=True).replace(':', '').strip()
                
    #             # Get the text from the main box, preserving line breaks
    #             text = text_tag.get_text(strip=True, separator='\n').strip()
                
    #             articles_data.append({'title': title, 'text': text})

    #     # Step 5: Write the data to a CSV file
    #     if not articles_data:
    #         print("No articles were extracted. The CSV file will be empty.")
    #         return

    #     with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
    #         fieldnames = ['title', 'text']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         writer.writerows(articles_data)

    #     print(f"Successfully scraped {len(articles_data)} articles into '{output_csv_path}'")

    # except FileNotFoundError:
    #     print(f"Error: The file '{html_content}' was not found.")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")