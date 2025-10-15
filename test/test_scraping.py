import csv
import requests
from bs4 import BeautifulSoup

def scrape_law_articles(html_content, output_csv_path):
    """
    Parses an HTML file to extract law articles and saves them to a CSV file.

    Args:
        html_file_path (str): The path to the input HTML file.
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Step 3: Find all article containers
        article_divs = soup.find_all('div', class_='article_item')

        articles_data = []
        print(f"Found {len(article_divs)} article containers to process.")

        # Step 4: Loop through each article and extract only the primary text
        for article in article_divs:
            title_tag = article.find('h3', class_='center')
            
            # Use .find() to get the FIRST 'HTMLContainer', which is the main text box.
            # This intentionally skips the amendment text in the nested popup.
            text_tag = article.find('div', class_='HTMLContainer')

            if title_tag and text_tag:
                title = title_tag.get_text(strip=True).replace(':', '').strip()
                
                # Get the text from the main box, preserving line breaks
                text = text_tag.get_text(strip=True, separator='\n').strip()
                
                articles_data.append({'title': title, 'text': text})

        # Step 5: Write the data to a CSV file
        if not articles_data:
            print("No articles were extracted. The CSV file will be empty.")
            return

        with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['title', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(articles_data)

        print(f"Successfully scraped {len(articles_data)} articles into '{output_csv_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{html_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


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
    output_csv = 'law_articles.csv'
    scrape_law_articles(input_html, output_csv)