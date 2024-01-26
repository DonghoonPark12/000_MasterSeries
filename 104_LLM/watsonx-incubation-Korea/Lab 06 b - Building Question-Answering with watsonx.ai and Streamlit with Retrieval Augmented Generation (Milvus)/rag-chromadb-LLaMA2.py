import os, requests
import pandas as pd
from dotenv import load_dotenv
from typing import Any, List, Mapping, Optional, Union, Dict

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Could not import sentence_transformers: Please install sentence-transformers package.")

# Vector DB    
try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction
except ImportError:
    raise ImportError("Could not import chromdb: Please install chromadb package.")
    
from rouge import Rouge


from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

def get_wml_creds():
    load_dotenv()
    api_key = os.getenv("API_KEY", None)
    ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
    project_id = os.getenv("PROJECT_ID", None)
    if api_key is None or ibm_cloud_url is None or project_id is None:
        print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
    else:
        creds = {
            "url": ibm_cloud_url,
            "apikey": api_key 
        }
    return project_id, creds

project_id, creds = get_wml_creds()

#--------------------------------------------------------------------------------#
# 1. 데이터 로드
def load_data_v1(data_dir, data_root):
    passages = pd.read_csv(os.path.join(data_dir, "passages.tsv"), sep='\t', header=0)
    qas = pd.read_csv(os.path.join(data_dir, "questions.tsv"), sep='\t', header=0).rename(columns={"text": "question"})
    
    # We only use 5000 examples.  Comment the lines below to use the full dataset.
    passages = passages.head(5000)
    qas = qas.head(5000)
    
    return passages, qas
data_root = './'
data_dir = os.path.join(data_root, 'LongNQ')
documents, questions = load_data_v1(data_dir, data_root)
documents['indextext'] = documents['title'].astype(str) + "\n" + documents['text']
#--------------------------------------------------------------------------------#

# 2. 임베딩 클래스 생성
"""
chromadb.api.types.EmbeddingFunction 를 상속하여 커스터한 임베딩 클래스 만들어 준다.
"""
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        # SentenceTransformer의 encode 메소드 호출 + text를 입력으로 받는다.
        #return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()
        return MODEL.encode(texts).tolist()
emb_func = MiniLML6V2EmbeddingFunction()

# 3. Chroma Upsert
'''
중복 되는 키 값이 있다면 해당 value를 업데이트하고,
그렇지 않다면(DB안에 없는 케이스라면) 해당 키 값에 대한 value를 추가
'''
class ChromaWithUpsert:
    def __init__(self, name, persist_directory, embedding_function,
                 collection_metadata: Optional[Dict] = None):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._name = name
        self._collection = self._client.get_or_create_collection(
            
        )

# 4. Chroma를 이용해서 임베딩 과 도큐먼트 인덱싱