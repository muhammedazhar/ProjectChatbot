
# AutoScraper

This tool is a planned feature for the [**ProjectChatbot**](https://github.com/muhammedazhar/ProjectChatbot) to ask the model about price of any specific product it is watching. It's a simple web scraper that can be used to extract data from websites in an simple way.
## Acknowledgements
I utilised freely available online resources from [autoscraper](https://pypi.org/project/autoscraper/) website and from [YouTube video](https://youtu.be/unPQEenF2aE?si=5iMWTslEBqzdjHrI) from Mark McNally.

 - [autoscraper PyPI - v1.1.14](https://pypi.org/project/autoscraper/1.1.14/)
 - [The EASIEST way to do WEB SCRAPING with PYTHON]()
 - [Industrial-scale Web Scraping with AI & Proxy Networks](https://www.youtube.com/watch?v=qo_fUjb02ns)

These are the resources I used to built this specific AutoScraper.
## Authors

- [@muhammedazhar](https://www.github.com/muhammedazhar)
## Running Tests

Follow these steps to run tests:

#### Step 1: Install Autoscraper

Before running tests, install the **`autoscraper`** tool by executing the following command in your terminal:

```bash
pip install autoscraper
```

#### Step 2: Train the Scraper

Once installed, use **`autoscraper`** to train the scraper based on the elements you identified. Execute the following command:

```bash
python build_price.py
```

#### Step 3: Retrieve Scraped Data

After training the scraper, an output will be generated. To retrieve the scraped result from the web, execute the following command:

```bash
python get_price.py
```

By following these steps, you'll be able to successfully run tests using the Autoscraper tool.
## License

This project is licensed under the [MIT License](../../docs/LICENSE).

## Questions and Support

If you have any questions or need support, feel free to open an issue or reach out to [Muhammed Azhar](https://github.com/muhammedazhar).

Happy coding!