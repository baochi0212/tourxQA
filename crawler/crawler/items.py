# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from operator import delitem
import scrapy
from scrapy.loader import ItemLoader
from typing import Dict

class CustomItem(scrapy.Item):
    # define the fields for your item here like:
    # override the data-type, the input
    fields: Dict[str, scrapy.Field]
    title = scrapy.Field()
    content = scrapy.Field()
    meta = scrapy.Field()
    def __init__(self, *args, **kwargs):
        self._values = dict([(key, None) for key in self.fields.keys()])
        if args or kwargs:  # avoid creating dict for most common case
            for k, v in dict(*args, **kwargs).items():
                self[k] = v
    def __getattr__(self, name):
        return self.name
    def setValue(self):
            self['title'] = self['meta']['title'] 
            self['content'] = self['meta']['content']
            del self['meta']
    def __setitem__(self, key, value):
        self._values[key] = value

class ChatbotItem(scrapy.Item):
    content = scrapy.Field()
    def preprocess(self):
        self['content'] = self['content'].lower()

if __name__ == "__main__":
    item = ChatbotItem(meta = {'title': 3, 'content': 4}, content='HOA')
    item.preprocess()
    print(item._values)
    