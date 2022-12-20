import os

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import convert_files_to_docs
from haystack.nodes import BM25Retriever    

#config host
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

if __name__ == "__main__":
    document_store = ElasticsearchDocumentStore(host=host, username="baochi2k2", password="Baochi2012", index="ngu")
    docs = convert_files_to_docs(dir_path='database/test', split_paragraphs=True)
    document_store.write_documents(docs)

    #retriever
    retriever = BM25Retriever(document_store=document_store)
    #query for candidates
    print("RESULT: ", retriever.retrieve(query="hấp dẫn", top_k=2))
    print("DOCS", docs)
