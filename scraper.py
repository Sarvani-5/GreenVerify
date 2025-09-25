import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urljoin, urlparse
import csv

class WebCrawler:
    def __init__(self, delay=1):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.delay = delay
        
    def get_page(self, url):
        """Fetch a single page with error handling"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
            
    def extract_text_content(self, soup):
        """Extract clean text content from soup object"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    
    def extract_links(self, soup, base_url):
        """Extract all links from the page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            links.append({
                'url': full_url,
                'text': link.get_text(strip=True),
                'title': link.get('title', '')
            })
        return links
    
    def extract_images(self, soup, base_url):
        """Extract all images from the page"""
        images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                full_url = urljoin(base_url, src)
                images.append({
                    'url': full_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
        return images
    
    def crawl_single_page(self, url):
        """Crawl a single page and extract information"""
        print(f"Crawling: {url}")
        
        response = self.get_page(url)
        if not response:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract basic information
        data = {
            'url': url,
            'title': soup.title.string if soup.title else 'No title',
            'meta_description': '',
            'headings': {},
            'links': self.extract_links(soup, url),
            'images': self.extract_images(soup, url),
            'text_content': self.extract_text_content(soup)
        }
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            data['meta_description'] = meta_desc.get('content', '')
        
        # Extract headings
        for i in range(1, 7):
            headings = soup.find_all(f'h{i}')
            data['headings'][f'h{i}'] = [h.get_text(strip=True) for h in headings]
        
        # Add delay to be respectful
        time.sleep(self.delay)
        
        return data
    
    def save_to_json(self, data, filename):
        """Save crawled data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")
    
    def save_to_csv(self, data, filename):
        """Save basic crawled data to CSV file"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['URL', 'Title', 'Meta Description', 'Text Length'])
            
            if isinstance(data, list):
                for item in data:
                    writer.writerow([
                        item.get('url', ''),
                        item.get('title', ''),
                        item.get('meta_description', ''),
                        len(item.get('text_content', ''))
                    ])
            else:
                writer.writerow([
                    data.get('url', ''),
                    data.get('title', ''),
                    data.get('meta_description', ''),
                    len(data.get('text_content', ''))
                ])
        print(f"CSV data saved to {filename}")

def main():
    # Example usage
    crawler = WebCrawler(delay=2)  # 2 second delay between requests
    
    # Single page crawling
    url = "https://www.grihaindia.org/griha-rating"
    page_data = crawler.crawl_single_page(url)
    
    if page_data:
        print(f"\nTitle: {page_data['title']}")
        print(f"Meta Description: {page_data['meta_description']}")
        print(f"Number of links: {len(page_data['links'])}")
        print(f"Number of images: {len(page_data['images'])}")
        print(f"Text content length: {len(page_data['text_content'])} characters")
        
        # Print headings
        print("\nHeadings found:")
        for heading_level, headings in page_data['headings'].items():
            if headings:
                print(f"{heading_level}: {headings}")
        
        # Save data
        crawler.save_to_json(page_data, 'crawled_data.json')
        crawler.save_to_csv(page_data, 'crawled_data.csv')
        
        # Print first 500 characters of text content
        print(f"\nFirst 500 characters of content:")
        print(page_data['text_content'][:500] + "...")

if __name__ == "__main__":
    main()