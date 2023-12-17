import os
import getpass
import tempfile

#import streamlit as st

from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import (HuggingFaceHubEmbeddings, HuggingFaceInstructEmbeddings)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['OPENAI_API_KEY'] = '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' #getpass.getpass('OpenAI API Key:')

"""
https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
"""
#----------------------------------------------------------------------------------------------------------#
# Document Loaders 
uploaded_files = './layout-parser-paper.pdf'

loader = PyPDFLoader(uploaded_files)
pages = loader.load_and_split() # 해당 방식은 문서를 페이지 번호로 검색할 수 있다.
pages_d = loader.load()
print(pages[0])
"""
Document(page_content='', type='Document', metadata={'source': './layout-parser-paper.pdf', 'page': 0}) 의
리스트가 담겨있다.
"""

loader2 = PyPDFLoader(uploaded_files, extract_images=True)
pages2 = loader2.load()
print(pages2[3].page_content)

#----------------------------------------------------------------------------------------------------------#
# Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 100,
#     chunk_overlap  = 20,
#     length_function = len,
#     add_start_index = True,
# )

# chunk_size = 1000
# chunk_overlap = 150
# r_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
# docs = r_splitter.split_documents(pages)
# print(docs)

# texts = text_splitter.create_documents([state_of_the_union])
# print(texts[0])
# print(texts[1])
#----------------------------------------------------------------------------------------------------------#

# faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
# docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

