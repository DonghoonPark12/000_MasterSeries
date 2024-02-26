import streamlit as st

#--------------------------------------------------------------
# title


#--------------------------------------------------------------
# sidebar
with st.sidebar:
    st.title(":blossom: Generative AI Interfaces")
    model_type_sel = st.radio(
        " 사용하실 기능을 선택해 주세요", # Please select the function you want to use
        ('text-to-image', 'text-to-video', 'LLM'),
        captions=["사용자가 입력한 텍스트로 부터 이미지를 생성합니다.",
                  "사용자가 입력한 텍스트로 부터 비디오를 생성합니다.",
                  "OpenAI ChatGPT"]
    )
    
    if model_type_sel == 'text-to-image': 
        option = st.selectbox(
            '사용하실 모델을 선택해주세요', # Please select the model you want to use
            ('None', 'Dalle2 (Not Implemented)', 'Stable-Diffusion (Not Implemented)', 'Imagen (Not Implemented)'),
            placeholder='None'
        )
    elif model_type_sel == 'text-to-video':
        option = st.selectbox(
            '사용하실 모델을 선택해주세요', # Please select the model you want to use
            ('None', "SoRa (Not Implemented)"),
            placeholder='None'
        )
    elif model_type_sel == 'LLM':
        pass

'https://github.com/CompVis/stable-diffusion'
'https://github.com/Stability-AI/generative-models'
'https://platform.stability.ai/sandbox/text-to-image'