from googlesearch import search
import scrapy

# to search
query = "Huế những địa điểm không thể bỏ qua"
website = " Kenh14"
links = []
for j in search(query+website, num_results=2):
    links.append(j)
    #print(j)
print(links)

#Scrapy
class TestSpider(scrapy.Spider):
    name = "content"
    start_urls = links

    def parse(self, response):
        self.logger.info('hello Im trying to get the content of online articles on the website kenh14')
        content = response.css('div.knc-content')
        for paragraph in content:
            yield {
                'paragraph-title': paragraph.xpath("//p/b/text()").getall(),
                'text': paragraph.xpath("//p/text()").getall(),
                'text-in-boxes': paragraph.xpath("//p/span/text()").getall(),
                'italic-text': paragraph.xpath("//p/i/text()").getall(),
                'text-links': paragraph.xpath("//a[@class='link-inline-content']/text()").getall(),
            }
        #print(content)
        pass