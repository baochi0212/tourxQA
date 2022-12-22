from glob import glob
import os
import json

from googlesearch import search


from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import convert_files_to_docs
from haystack.nodes import BM25Retriever
#directory
working_dir = os.environ['dir']
database_dir = f"{working_dir}/data/database"


# # Get the host where Elasticsearch is running, default to localhost
# host = os.environ.get("ELASTICSEARCH_HOST", "localhost")
# document_store = ElasticsearchDocumentStore(host=host, username="baochi0212", password="Baochi2002", index="document")


# retriever = BM25Retriever(document_store=document_store)
# #add_dcuments
# # Now, let's write the docs to our DB.
# docs = convert_files_to_docs(dir_path=database_dir, split_paragraphs=True)
# document_store.write_documents(docs)




class Crawl:
    def __init__(self, bot_name="kenh14"):
        self.name = bot_name
        #urls for bot to scrape
        if not os.path.exists(f"{database_dir}/test/urls.txt"):
            open(f"{database_dir}/test/urls.txt", "w")
    def json2txt(self):
        files = []
        for file in glob(f"{database_dir}/test/*.json"):
            with open(file, 'r') as f:
                files.extend(json.load(f))
        #write to text file
        for i, line in enumerate(files):
            with open(f'{database_dir}/test/docs/text_{i}.txt', 'w') as f:
                f.write(line['content'])
    def query(self, q="mon an ngon Da Nang", num_results=3):
        for i, result in enumerate(search(q + " " + self.name, num_results=num_results)):
            with open(f"{database_dir}/test/urls.txt", "a") as f:
                f.write(result + '\n')

        self.crawl()
        self.json2txt()
                
            
    def crawl(self):
        current_docs = len(glob(f"{database_dir}/test/*.json"))
        print("num", current_docs)
        os.system(f"scrapy runspider crawler/spiders/{self.name}.py -o {database_dir}/test/text{current_docs + 1}.json")
        
        
        #reset url-file after one crawling session
        open(f"{database_dir}/test/urls.txt", "w")






if __name__ == "__main__":
    # Crawl("kenh14").query(num_results=3)
    q="mon an ngon Da Nang" 
    num_results=3
    for i, result in enumerate(search(q + " " + 'kenh14', num_results=num_results)):
            with open(f"{database_dir}/test/urls.txt", "w") as f:
                f.write(result + '\n')
