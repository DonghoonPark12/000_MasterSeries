import streamlit as st

from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Web Search
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

from langchain.callbacks import StreamlitCallbackHandler
"""
[원칙]
- app.py에서는 OpenAI, LangChain 사용이 어떻게 되는지 모르게 한다. 
"""

def generate_response(input_text, temperature, openai_api_key):
    llm = OpenAI(temperature=temperature, openai_api_key=openai_api_key)
    output_text = llm(input_text)
    return output_text

def generate_response_new(input_text, openai_api_key):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)



# 현재 대한민국 대통령 누구야?