import scrapy


class QizhaSpider(scrapy.Spider):
    name = 'qizha'
    allowed_domains = ['tieba.baidu.com']
    start_urls = ['https://tieba.baidu.com/f?fr=wwwt&kw=%E9%98%B2%E6%AC%BA%E8%AF%88/']

    def parse(self, response):
        pass
