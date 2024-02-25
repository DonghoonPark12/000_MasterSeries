'''
st.sidebar
- 위젯을 사용하여 앱에 상호작용 기능을 추가할 수 있을 뿐만 아니라 위젯을 사이드바로 정리할 수도 있습니다.
- 아래처럼 객체 표기법을 이용하여 st.sidebar.{}로 추가할 수도 있고, with st.sidebar: 를 이용할 수 도 있다.

- 객체 표기법을 사용하여 지원되지 않는 유일한 요소는 st.echo, st.spinner 및 st.toast입니다. 
- 이러한 요소를 사용하려면 with st.sidebar: 표기법과 함께 사용해야 합니다.

https://docs.streamlit.io/library/api-reference/layout/st.sidebar
'''

import streamlit as st
import time
# 아래 3가지 방식은 모두 동일
# 1) 
add_sidebar = st.sidebar.link_button(
    'Google Translation', 'https://translate.google.com/'
)

# 2)
st.sidebar.link_button(
    'Google Translation', 'https://translate.google.com/'
)

add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# 3) 하나로 모아서 코드를 짤 수 있어서 이 방법이 편리한 듯??
# st.selectbox 이후 첫 인자인 label이 동일하면 하나만 뜬다.
with st.sidebar:
    st.link_button(
    'Google Translation', 'https://translate.google.com/'
    )
    st.selectbox(
    "How would you like to be contacted? (2)",
    ("Email", "Home phone", "Mobile phone")
    )
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )

# picture = st.camera_input("Take a picture")

# if picture:
#     st.image(picture)

# st.sidebar.echo("This code will be printed to the sidebar.(2)") # 이게 안된다.
#st.sidebar.spinner("Loading") # 이건, Loading 표현을 하기 위해서 with notation과 써야만 하는 것이 이해가 된다.

with st.sidebar:
    with st.echo():
        #st.write("This code will be printed to the sidebar.")
        #st.text("This code will be printed to the sidebar.")
        "This code will be printed to the sidebar."

    with st.spinner("Loading..."):
        time.sleep(5)
    st.success("Done!") # 단순 초록색 표시
    



    
