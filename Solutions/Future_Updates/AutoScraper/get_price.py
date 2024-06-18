from autoscraper import AutoScraper

scraper = AutoScraper()

scraper.load('price_scraper')

def get_price_value(url):
    return scraper.get_result_exact(url)

prices = [
    'apple-iphone-14-pro-max-deep-purple-128-gb/p/itm5256789ae40c7',
    'apple-iphone-14-pro-deep-purple-128-gb/p/itm75f73f63239fa',
    'google-pixel-7-pro-obsidian-128-gb/p/itmb74dc5c3b3eb5',
    'google-pixel-7-snow-128-gb/p/itm45d75002be0e7'
]


product_url = {
    'apple-iphone-14-pro-max-deep-purple-128-gb/p/itm5256789ae40c7',
    'apple-iphone-14-pro-deep-purple-128-gb/p/itm75f73f63239fa',
    'google-pixel-7-pro-obsidian-128-gb/p/itmb74dc5c3b3eb5',
    'google-pixel-7-snow-128-gb/p/itm45d75002be0e7'
}
product = {
    'Apple iPhone 14 Pro Max',
    'Apple iPhone 14 Pro',
    'Google Pixel 7 Pro',
    'Google Pixel 7'
}

for price in prices:
    url = f'https://www.flipkart.com/{price}?'
    price_value = get_price_value(url)
    price_of_product = f'As of now the {price} is priced at {price_value}'

for value in product_url:
        if value in price_of_product:
            index = list(product_url).index(value)
            replacement = list(product)[index]
            price_of_product = price_of_product.replace(value, replacement)

print(price_of_product)