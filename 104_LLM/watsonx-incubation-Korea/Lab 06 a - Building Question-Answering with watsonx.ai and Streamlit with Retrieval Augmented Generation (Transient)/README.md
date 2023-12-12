# 벡터 DB를 활용한 LLM 기반 QA 앱 만들기
이번 시간에는 하나의 문서 안에서 중요한 정보를 추출할 수 있는 Vector DB를기반으로 LLM 앱을 실행할 수 있는 방법론에 대해 배워봅니다. 이를 통해 LLM 앱은 기업 내 중요 문서에 있는 정보를 추가적인 학습 없이 효율적으로 사용할 수 있어 보안성 유지와 함께 사용자에게 효과적인 정보를 전달할 수 있습니다.

# 허깅페이스(HuggingFace) API Setup
1. 아래 사이트에 로그인 하거나 가입하세요 https://huggingface.co/.
2. 세팅 메뉴를 클릭하세요 "Settings".
3. "Access Token"에 가서 "New token"을 클릭하세요.
4. 토큰에 이름을 부여하고 다음과 같이 권한을 부여하세요 permission=write. 그리고 토큰을 생성하세요
5. 해당 토큰을 .env file에 아래와 같이 추가하세요. `HUGGINGFACEHUB_API_TOKEN=<your_new_huggingface_access_token>`

# 실행 방법 🚀
1. 자신의 컴퓨터 터미널 혹은 콘솔을 시작하세요
2. 가상환경안에 있는지 확인하세요
3. 우리가 실행할 코드 (Lab 6a - Building Question-Answering with watsonx.ai and Streamlit with Retrieval Augmented Generation (Transient))로 이동하세요
4. .env 파일을 해당 폴더 안에 복사하세요
5. 다음 명령어를 입력하여 앱을 사용 해보세요 `streamlit run app.py`

### 질문 예시: [To be chaned]
-  쉬운 질문
    - 직원의 정규 근무시간은?
    - 직원의 정년은 언제입니까?
    - 신입 사원의 온보딩 기간은 얼마나 됩니까?
    - 경력직의 온보딩 기간은 얼마나 됩니까?
    - 직원이 사망하면 어떻게 보상 받나요?
    
- 구체적 질문
    - 급여일이 토요일인 경우에는 언제 급여가 지급되나요?
    - 직원의 요일별 복장 규정에 대해 설명해주세요
    - 휴가를 사용하지 않았는데 휴일 수당을 받을 수 있나요?
    
