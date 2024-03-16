import os
import streamlit as st
from models import generate_response, generate_response_new


#--------------------------------------------------------------
# title


#--------------------------------------------------------------
# global variables
chatgpt_flag = False

#--------------------------------------------------------------
# functions
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}]

#--------------------------------------------------------------
# sidebar
with st.sidebar:
    st.title(":blossom: Generative AI Interfaces")
    model_type_sel = st.radio(
        " 사용하실 기능을 선택해 주세요", # Please select the function you want to use
        ('text-to-image', 'text-to-video', 'LLM'),
        captions=["사용자가 입력한 텍스트로 부터 이미지를 생성합니다.",
                  "사용자가 입력한 텍스트로 부터 비디오를 생성합니다.",
                  "사용자가 입력한 텍스트로 부터 텍스트를 생성합니다."]
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
        option = st.selectbox(
            '사용하실 모델을 선택해주세요', # Please select the model you want to use
            ('None', "Llama2 (Not Implemented)", "ChatGPT", "Gemma (Not Implemented)"),
            placeholder='None'
        )
        temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=5.0, value=0.7, step=0.1)
        #top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        #max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
        if option == "Llama2":
            #------------------------------------------------------------------------------#
            if 'REPLICATE_API_TOKEN' in st.secrets:
                st.success('API key already provided!', icon='✅')
                replicate_api = st.secrets['REPLICATE_API_TOKEN']
            else:
                replicate_api = st.text_input('Enter Replicate API token:', type='password')
                if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
                    st.warning('Please enter your credentials!', icon='⚠️')
                else:
                    st.success('Proceed to entering your prompt message!', icon='👉')
            os.environ['REPLICATE_API_TOKEN'] = replicate_api

            st.subheader('Models and parameters')
            selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')

            if selected_model == 'Llama2-7B':
                llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
            elif selected_model == 'Llama2-13B':
                llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
            #------------------------------------------------------------------------------#

        elif option=="ChatGPT":
            openai_api_key = st.text_input('OpenAI API Key', type='password')
            chatgpt_flag = True

            st.button('Clear Chat History', on_click=clear_chat_history)

if chatgpt_flag == True:
    # -------------------------------------------------------------------------------#
    # 참조: https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/
    # # Store LLM generated responses # ??
    # if "messages" not in st.session_state:
    #     st.session_state["messages"] = [
    #         {"role": "assistant", 
    #          "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    #         ]

    # # Display or clear chat messages
    # for msg in st.session_state.messages:
    #     st.chat_message(msg["role"]).write(msg["content"]) # role 과 content???

    # # User-provided prompt
    # if prompt := st.chat_input():
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     st.chat_message("user").write(prompt)

    #     if not openai_api_key.startswith('sk-'):
    #         st.warning('Please enter your OpenAI API key!', icon='⚠')
    #         st.stop()

    #     generate_response_new(prompt, openai_api_key)
    
    # -------------------------------------------------------------------------------#
    # https://docs.streamlit.io/knowledge-base/tutorials/llm-quickstart
    with st.form('my_form'):  #[TODO] 포맷이 구리다. 교체하기
        input_text = st.text_area('Enter text: ')
        submitted = st.form_submit_button('Submit')
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and openai_api_key.startswith('sk-'):
            output_text = generate_response(input_text, temperature, openai_api_key)
            st.info(output_text)
    # -------------------------------------------------------------------------------#
            

#'https://github.com/CompVis/stable-diffusion'
#'https://github.com/Stability-AI/generative-models'
#'https://platform.stability.ai/sandbox/text-to-image'