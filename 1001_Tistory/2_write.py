import requests
import json
from pprint import pprint


import os
from dotenv import load_dotenv # 환경 변수 관리 프로그램
load_dotenv()

REDIRECT_URI = os.getenv("REDIRECT_URI", None)
CODE = os.getenv("CODE", None)
CLIENT_SECRET = os.getenv("CLIENT_SECRET", None)
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", None)
CLIENT_ID = os.getenv("PROJECT_ID", None)
BLOGNAME = os.getenv("BLOG_NAME", None)

def getAccessToken():
    url = "https://www.tistory.com/oauth/access_token?"
    client_id = CLIENT_ID
    client_secret = CLIENT_SECRET
    code = CODE
    redirect_uri = REDIRECT_URI
    grant_type="authorization_code" # authorization_code 고정

    data = url
    data += "client_id="+client_id+"&"
    data += "client_secret="+client_secret+"&"
    data += "redirect_uri="+redirect_uri+"&"
    data += "code="+code+"&"
    data += "grant_type="+grant_type
    print(data)
    return  requests.get(data)

def getCategoryID():
    """
    https://tistory.github.io/document-tistory-apis/apis/v1/post/list.html
    """
    url = "https://www.tistory.com/apis/category/list?"
    output = "json"
    blogName = BLOGNAME

    data = url
    data += "access_token=" + ACCESS_TOKEN + "&"
    data += "output=" + output + "&"
    data += "blogName=" + blogName

    print(data)
    return requests.get(data)

def postWriting(title="No Title", content="No Content", id="", tag=""):
    url = "https://www.tistory.com/apis/post/write?"
    output = "json"
    blogName = "dhpark-codereview"

    data = url
    data += "access_token=" + ACCESS_TOKEN + "&"
    data += "output=" + output + "&"
    data += "blogName=" + blogName + "&"
    data += "title=" + title + "&"
    data += "content=" + content + "&"
    data += "visibility=3&"
    data += f"category={id}"
    #data += f"tag={tag}"

    print(data)
    return requests.post(data)

if __name__ == "__main__":
    #token = getAccessToken().content
    #print(token.decode('utf-8'))

    # category = json.loads(getCategoryID().content)
    # category = json.dumps(category, indent=4, ensure_ascii=False)
    # pprint(category)
    # with open("./category.json", "w") as outfile:
    #     outfile.write(category)

    title = "Mitigating the Effect of Incidental Correlations on Part-based Learning"
    contents = "한글 제목: 부분 기반 학습에서 우발적 상관관계의 영향 완화 요약: 이 논문에서는 부분 기반 학습에서 나타나는 우발적 상관관계 문제를 해결하기 위해 분리된 부분 기반 비전 트랜스포머(DPViT)를 소개합니다. DPViT는 독특한 부분 혼합 방식과 자기 감독 규제를 사용하여 전경과 배경 정보의 생성 과정을 분리합니다. 이 방법은 희소성 및 스펙트럼 직교 제약을 적용하여 부분 표현의 고품질을 보장합니다. 연구는 DPViT가 소수샷 학습 작업에서 최신 성능을 달성하고 도메인 변화 및 데이터 손상을 효과적으로 처리함을 보여줍니다. 제안된 방법의 효과는 MiniImageNet, TieredImageNet, FC100 등 다양한 벤치마크 데이터셋에서 검증되었습니다. 자세한 그림과 표는 원본 논문에서 확인할 수 있습니다: GitHub URL (https://github.com/GauravBh1010tt/DPViT.git). 논문 전문 PDF 링크: 논문 PDF 다운로드 (https://ar5iv.labs.arxiv.org/html/2310.00377)"   

    # w1 = postWriting(title=title, content=contents, id="919344").content
    # posting = json.loads(w1)
    # print(posting)