# 가상환경 만들기
### Python 설치
homebrew를 활용하여 Python을 설치하고 관리해보겠습니다.  
##### 아래 명령어를 차례대로 입력하세요:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.11
brew install git
```
설치가 되었다면, [교육자료](https://github.com/Yupjun/watsonx-incubation-program-Korea)를 다운로드 받고 해당 경로로 이동합니다.  
###### 아래 명령어를 차례대로 입력하세요:
```
git clone https://github.com/Yupjun/watsonx-incubation-program-Korea
cd watsonx-incubation-program-Korea
python -m venv .venv

```

#### 가상환경을 실행하여 가상환경 안으로 들어갑니다
##### 아래 명령어를 차례대로 입력하세요:
```
source .venv/bin/activate
python -m pip install -r requirements_venv.txt
```
터미널 옆에 (venv) 라는 형태로 표시가 되어 있다면, 제대로 진행된 것입니다.

#### 주피터 노트북을 실행하여 테스트합니다.
이제 주피터 노트북을 실행하고, lab-0-laptop-environment-setup에 있는 environment-test.ipynb를 실행하여   
모든 패키지가 제대로 작동하는지 확인합니다. Cloud 환경 세팅의 경우 [링크](./Setting-up-Python-Virtual-Environment-in-Windows.pdf) 내 Cloud 환경 세팅 란을 참고하여 진행해주세요
##### 아래 명령어를 차례대로 입력하세요:
```
jupyter notebook
```
#### jupyter notebook에서의 테스트가 끝났다면, [다음링크](https://docs.docker.com/desktop/install/mac-install/)를 통해 Docker를 설치합니다.

# 끝
