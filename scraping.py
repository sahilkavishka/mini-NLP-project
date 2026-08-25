import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# 1. Setup Robust Session with Retries
def get_secure_session():
    session = requests.Session()
    
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# 2. Random User-Agents to prevent blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
]

def scrape_professional_dataset(phone_urls, pages_per_phone=10):
    session = get_secure_session()
    all_data = []
    
 
    for phone_url in phone_urls:
        print(f"\nScraping Device: {phone_url.split('/')[-1].split('-')[0]}")
        base_url = phone_url.replace(".php", "")
        
       
        for page in tqdm(range(1, pages_per_phone + 1), desc="Pages Scraped"):
            url = f"{base_url}.php" if page == 1 else f"{base_url}p{page}.php"
            
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            
            try:
                response = session.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # GSMArena review blocks (user-thread)
                review_blocks = soup.find_all('div', class_='user-thread')
                
                for block in review_blocks:
                    # User Name
                    uname_tag = block.find('li', class_='uname')
                    username = uname_tag.get_text(strip=True) if uname_tag else "Anonymous"
                    
                    # Date
                    date_tag = block.find('li', class_='upost')
                    post_date = date_tag.get_text(strip=True) if date_tag else None
                    
                    # Review Text
                    text_tag = block.find('p', class_='uopin')
                    review_text = text_tag.get_text(strip=True) if text_tag else None
                    
                    # Rating/Upvotes (If available in standard format)
                    upvote_tag = block.find('span', class_='thumbs-up')
                    upvotes = upvote_tag.get_text(strip=True) if upvote_tag else "0"
                    
                    if review_text:
                        all_data.append({
                            'device_url': base_url,
                            'username': username,
                            'date': post_date,
                            'upvotes': upvotes,
                            'review_text': review_text
                        })
                        
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                
           
            time.sleep(random.uniform(2.0, 4.5)) 

    # Data Quality Check & Save
    df = pd.DataFrame(all_data)
    
   
    df.dropna(subset=['review_text'], inplace=True)
    df.drop_duplicates(subset=['review_text'], inplace=True)
    
    filename = 'gsmarena_professional_dataset.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\nSuccessfully saved {len(df)} unique reviews to '{filename}'!")
    return df

# -- Execution --

target_devices = [
    # 1. Flagship Phones (Premium level - High expectations)
    "https://www.gsmarena.com/samsung_galaxy_s24_ultra-reviews-12771.php",
    "https://www.gsmarena.com/apple_iphone_15_pro_max-reviews-12548.php",
    
    # 2. Mid-Range Phones (Balanced reviews)
    "https://www.gsmarena.com/samsung_galaxy_a54-reviews-12070.php",
    "https://www.gsmarena.com/xiaomi_redmi_note_13_pro-reviews-12581.php",
    
    # 3. Budget / Entry-Level Phones (More critical/negative reviews)
    "https://www.gsmarena.com/nokia_g22-reviews-12145.php",
    "https://www.gsmarena.com/realme_c55-reviews-12159.php",

    # 4. Older/Controversial Phones (To get more mixed sentiments)
    "https://www.gsmarena.com/apple_iphone_se_(2022)-reviews-11410.php"
]

final_dataset = scrape_professional_dataset(target_devices, pages_per_phone=15)


print(final_dataset.head())
print("\nDataset Info:")
print(final_dataset.info())
