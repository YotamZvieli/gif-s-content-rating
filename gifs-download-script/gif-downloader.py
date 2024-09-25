import requests
import urllib.request
import os
import time

# Replace YOUR_API_KEY with your actual Giphy API key
API_KEY = 'ZobQmFXUCpag6irj5kw9ib6Tk89sGlF1'
BASE_URL = 'https://api.giphy.com/v1/gifs/'
LIMIT = 50
MAX_ATTEMPTS = 10
RATING_LIST = ['g', 'pg', 'pg-13', 'r']
BULK_LIMIT = 500
REQUEST_LIMIT = 100
TIME_TO_SLEEP = 3660
MAX_GIFS_FOR_RATING = 2000

def get_categories():
    params = {'api_key': API_KEY}
    response = requests.get(BASE_URL + 'categories', params)
    data = response.json()
    categories = []
    for category in data['data']:
        for subcategory in category['subcategories']:
            categories.append(subcategory['name'])

    return categories

def download_gifs(rating, type, params, num_to_download=LIMIT):

    response = requests.get(BASE_URL + type, params=params)
    data = response.json()
    folder_path = rf'downloaded_gifs\{rating}'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    files_in_folder = os.listdir(folder_path)
    downloaded_gifs_num = 0
    for gif in data['data']:
        
        file_name = f"{gif['title']}.gif"

        if rating == gif['rating'] and file_name not in files_in_folder:
            gif_url = gif['images']['original']['url']
        
            file_path = rf'{folder_path}\{file_name}'

            for j in range(MAX_ATTEMPTS):
                try:
                    urllib.request.urlretrieve(gif_url, file_path)
                    files_in_folder.append(file_name)
                    downloaded_gifs_num += 1
                    print(f'Downloaded: {file_name}')
                    if num_to_download <= downloaded_gifs_num:
                        return downloaded_gifs_num
                    break
                except:
                    print(f'failed the {j+1} attempt to download {file_name} in {rating} rating')
                    continue

    return downloaded_gifs_num

def download_trending(rating, request_count):
    downloaded_gifs_num = 0
    for i in range(0, BULK_LIMIT, LIMIT):
        if request_count == REQUEST_LIMIT:
            print('sleeping...')
            time.sleep(TIME_TO_SLEEP)
            request_count = 0
        
        params = {
            'api_key': API_KEY,
            'offset': i,
            'limit': LIMIT,
            'rating': rating
        }
        downloaded_gifs_num += download_gifs(rating, 'trending', params)
        request_count += 1

    return request_count, downloaded_gifs_num

def download_query(rating, category, request_count, num_to_download):
    downloaded_gifs_num = 0
    for i in range(0, BULK_LIMIT, LIMIT):
        if request_count == REQUEST_LIMIT:
            print('sleeping...')
            time.sleep(TIME_TO_SLEEP)
            request_count = 0
        
        params = {
            'api_key': API_KEY,
            'q': category,
            'offset': i,
            'limit': LIMIT,
            'rating': rating
        }
        
        downloaded_gifs_num += download_gifs(rating, 'search', params, num_to_download)
        request_count += 1
        if num_to_download <= downloaded_gifs_num:
            return request_count, downloaded_gifs_num
        num_to_download -= downloaded_gifs_num

    return request_count, downloaded_gifs_num

if __name__ == '__main__':
    categories = get_categories()
    request_count = 2 # Need to change to 1
    for rating in RATING_LIST:
        downloaded_gifs_num = 0
        print(f"Downloading trending gifs for {rating}")
        request_count, trends_downloaded_gifs_num = download_trending(rating, request_count)
        downloaded_gifs_num += trends_downloaded_gifs_num
        print(f"{trends_downloaded_gifs_num} trending gifs downloaded for {rating}")

        print(f"Downloading category gifs for {rating}")
        for i, category in enumerate(categories):
            num_to_download_per_category = (MAX_GIFS_FOR_RATING - downloaded_gifs_num) // (len(categories) - i)
            request_count, search_downloaded_gifs_num = download_query(rating, category, request_count, num_to_download_per_category)
            print(f"{search_downloaded_gifs_num} {category} gifs downloaded for {rating}")
            downloaded_gifs_num += search_downloaded_gifs_num

        print(f"{downloaded_gifs_num} gifs downloaded for {rating}")
