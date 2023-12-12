# Building  Apps with watsonx.ai and Streamlit
6a 세션에서는 벡터 DB를 활용하여 하나의 문서 만을 기반으로 RAG(Retrieval-Augmented Generation)를 하였습니다. LLM 모델 특성상 새로운 학습에는 굉장히 많은 시간과 비용이 소요되기 때문에, LLM 모델 하나만 사용하는 경우 새로운 정보를 즉시에 적용하기 어렵습니다. 이번 시간에는 하나의 문서 뿐만 아니라, 많은 문서들 중에서 중요한 정보를 기반으로 LLM 앱을 실행할 수 있는 방법론(Vector Search)에 대해 배워봅니다. 이를 통해 LLM 앱은 기업 내 중요 문서에 있는 정보를 추가적인 학습 없이 효율적으로 사용할 수 있어 보안성 유지와 함께 사용자에게 효과적인 정보를 전달할 수 있습니다. 

# Startup 🚀

1. 자신의 컴퓨터 터미널 혹은 콘솔을 시작하세요
2. 가상환경 안에 있는지 확인하세요
3. .env 파일을 실행할 폴더 내에 복사하세요
4. 설치된 도커 버전이 19.0 보다 높은지 확인합니다. (cmd: docker -v)
5. 도커가 설치되어 있다면 milvus directory 로 이동합니다(cd milvus). 그리고 다음 명령어를 입력하세요 "sudo docker-compose -f docker-compose.yml up" 도커가 설치되어 있지 않은 경우 rag-chromadb-LLaMA2.ipynb 만 진행 가능합니다. chromadb만 사용하는 경우 "rag-chromadb-LLaMA2.ipynb" 만 진행해주세요
6. http://localhost:8000/ 에 접속해서 milvus가 로드 되었는지 확인하세요
7.  wikihow.csv 파일을 다음 링크를 클릭하여 다운로드 받으세요 [LINK](https://ibm.box.com/s/4brucknqgrqahvbymatrce5zimiwfamj)  
8. milvus-demo.ipynb 를 차례대로 실행하여 wikihow.csv 파일을 milvus db에 로드하세요
10. `streamlit run app.py`을 실행합니다.
11. RAG를 통해 위키 데이터를 참고하는 LLaMA2 모델을 체험해봅시다. 아래는 예시 질문입니다

# 질문:
- 나무에 안전하게 오르기 위해서는 어떻게 해야 하나요?
- 자기소개서에 자신의 경험과 능력을 어떻게 말하면 좋나요?
- 비즈니스 자기소개서를 효과적으로 작성하는 방법과 그러한 자기소개서에서 피해야 할 사항은 무엇입니까?
- Lebesgue integral이란 무엇인가요?
