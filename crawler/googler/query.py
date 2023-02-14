from glob import glob
import os
import json
import argparse
from googlesearch import search


# from haystack.document_stores import ElasticsearchDocumentStore
# from haystack.utils import convert_files_to_docs
# from haystack.nodes import BM25Retriever
#directory
working_dir = os.environ['dir']
database_dir = f"{working_dir}/data/database"
parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, default='đà nẵng có món gì ngon ?')

# # Get the host where Elasticsearch is running, default to localhost
# host = os.environ.get("E`L`ASTICSEARCH_HOST", "localhost")
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
        def parse(string):
            new_string = ""
            start = True
            a, b = 1, 1
            while a != -1 and b != -1:
                a, b = string.find('<'), string.find('>')
            
                c = b + string[b+1:].find('<')
                if a != 0 and start:
                    new_string = string[:a]
                    start = False
                if c > 0:
                    join_char = ' '
                    new_string = join_char.join([new_string.strip(), string[b+1:c].strip()])
                else:
                    join_char = ' '
                    new_string = join_char.join([new_string.strip(), string.strip()])
                string = string[b+1:]
            return new_string.strip()
        files = []
        for file in glob(f"{database_dir}/test/*.json"):
            with open(file, 'r') as f:
                files.extend(json.load(f))
        f_write = open(f'{database_dir}/test/docs.txt', 'w')
        
        #write to text file
        for i, line in enumerate(files):
            text = parse(line['content'])
            f_write.write(text + '\n')
        f_write.close()

        #format the document
        f_read = open(f'{database_dir}/test/docs.txt', 'r')
        lines = f_read.readlines()
        f_write = open(f'{database_dir}/test/docs.txt', 'w')
        for line in lines:
            if not (line.startswith('Điện thoại') or line.startswith('Kenh14')):
                print(line)
                f_write.write(line.strip() + '\n')

        f_write.close()

        
    def query(self, q="Da Nang co mon gi ngon", num_results=3):
        #save urls queried
        for i, result in enumerate(search(q + " " + self.name, num_results=num_results)):
            #only .chn 
            if result[-3:] == 'chn':
                with open(f"{database_dir}/test/urls.txt", "a") as f:
                    f.write(result + '\n')

        #crawl
        # self.crawl()
        # self.json2txt()
                
            
    def crawl(self):
        current_docs = len(glob(f"{database_dir}/test/*.json"))
        print("num", current_docs)
        os.system(f"scrapy runspider crawler/spiders/{self.name}.py -o {database_dir}/test/text{current_docs + 1}.json")
        
        
        #reset url-file after one crawling session
        open(f"{database_dir}/test/urls.txt", "w")






if __name__ == "__main__":
    args = parser.parse_args()
    Crawl("kenh14").query(q=args.query, num_results=1)
    # q="mon an ngon Da Nang" 
    # num_results=3
    # for i, result in enumerate(search(q + " " + 'kenh14', num_results=num_results)):
    #         with open(f"{database_dir}/test/urls.txt", "w") as f:
    #             f.write(result + '\n')
  