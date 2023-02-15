import scrapy
from scrapy_splash import SplashRequest 

import os
import pandas as pd
import argparse
import sys
parser = argparse.ArgumentParser()
working_dir = os.environ['dir']
database_dir = f"{working_dir}/data/database"
crawler_dir = working_dir + "/crawler/crawler"
sys.path.append(crawler_dir)
from crawler.items import ChatbotItem
from crawler.utils import parse_json, parse_csv




lua_script = """
        function main(splash)
            local num_scrolls = 20
            local scroll_delay = 1.0

            local scroll_to = splash:jsfunc("window.scrollTo")
            local get_body_height = splash:jsfunc(
                "function() {return document.body.scrollHeight;}"
            )
            local url = splash.args.url
            assert(splash:go(url))
            splash:wait(splash.args.wait)

            for _ = 1, num_scrolls do
                scroll_to(0, get_body_height())
                splash:wait(scroll_delay)
            end        
            return {
                html = splash:html(),
                url = splash:url()
                }
                
        end



        """

class TestSpider(scrapy.Spider):
    name = 'test'
    def start_requests(self):
        urls = [url.strip() for url in open(f"{database_dir}/test/urls.txt", 'r').readlines()]
        for url in urls:
            yield SplashRequest(
                url, 
                callback=self.parse_article, 
                args={'wait': 2, 'lua_source': lua_script}
                )
    #parse the content of the articel
    def parse_article(self, response):
        #remove and replace word list
        remove_words = "\/"
        replace_words = '"'
        #cases handling
        content = ' '.join([text.split('>')[1].split('<')[0].strip() for text in response.css('p').css('[id^="docs-internal"]').getall()])
        if len(content.strip()) == 0:
            content = ' '.join([text.split('>')[1].split('<')[0].strip() for text in response.css('p').getall()])
        #remove unnecessary
        for word in remove_words:
            while word in content:
                content = content.replace(word, '')
        for word in replace_words:
            while word in content:
                content = content.replace(word, "'")
        #item yield
        item = ChatbotItem()
        item['content'] = content
        yield item
        



    # def parse(self, response):
    #     urls = []
    #     responses = response.css('.knswli-title')
    #     if len(responses) > 0:
            
    #         for r in responses:
    #             temp_item = {}
    #             url = r.css('a::attr(href)').get()
    #             temp_item['title_raw'] = url
    #             temp_item['url'] = "https://kenh14.vn/" + url

    #             urls.append(temp_item)

    #         # keep passing the metadata and used parser to next request
    #     df = pd.DataFrame.from_dict(dict([(k, [None]) for k in temp_item.keys()]))

        
    #     while len(df) != len(urls):  
            
    #         self.logger.info("NOT FINISHED YET")          
    #         for item in urls:
    #             if item['title_raw'] not in df['title_raw']:

    #                 yield SplashRequest(
    #                     item['url'], 
    #                     callback=self.parse_article, 
    #                     endpoint='execute', 
    #                     args={'wait': 0.5, 'lua_source': lua_script, 'meta': item} 
    #                     )
    #         df = pd.read_csv(test_path)
    #     # #trace
    #     # while len(pd.read_csv(test_path)) != len(urls):
    #     #     print("?????")
    #     #     self.logger.info("NOT FINISHED YET")
    #     #     df = pd.read_csv(test_path)
    #     #     for item in urls:
    #     #         if item['title_raw'] not in df['title_raw']:
    #     #             yield SplashRequest(
    #     #                 item['url'], 
    #     #                 callback=self.parse_article, 
    #     #                 endpoint='execute', 
    #     #                 args={'wait': 0.5, 'lua_source': lua_script, 'meta': item})
    #     # print("FINISHED ?")