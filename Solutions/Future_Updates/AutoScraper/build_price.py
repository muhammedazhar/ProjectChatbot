from autoscraper import AutoScraper

url = 'https://www.flipkart.com/apple-iphone-14-pro-max-deep-purple-128-gb/p/itm5256789ae40c7'

wanted_list = ['₹1,27,999']

scraper = AutoScraper()
scraper.build(url, wanted_list)

print(scraper.get_result_exact('https://www.flipkart.com/apple-iphone-14-pro-max-deep-purple-128-gb/p/itm5256789ae40c7'))

scraper.save('./price_scraper')