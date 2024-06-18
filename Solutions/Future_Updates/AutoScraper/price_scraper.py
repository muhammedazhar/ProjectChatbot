from autoscraper import AutoScraper
from flask import Flask, request

scraper = AutoScraper()

scraper.load('price_scraper')

app = Flask(__name__)

def get_prices(price):
    url = f'https://www.flipkart.com/{price}?'
    result = scraper.get_result_exact(url)
    return result

@app.route('/', methods=['GET'])
def search_api():
    query = request.args.get('price')
    return dict(result=get_prices(query))

if __name__ == '__main__':
    app.run(port=8080, host='0.0.0.0')