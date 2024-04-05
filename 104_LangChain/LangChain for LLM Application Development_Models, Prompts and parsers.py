import os
from pprint import pprint 

#import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
OpenAI.api_key = os.environ['OPENAI_API_KEY']

# account for deprecation of LLM model
import datetime
current_date = datetime.datetime.now().date() # Get the current date
target_date = datetime.date(2024, 6, 12) # Define the date after which the model should be set to "gpt-3.5-turbo" 

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

#--------------------------------------------------------------------------------#
# LangChain 없이 OpenAI API 만 사용하기
def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    #response = openai.ChatCompletion.create(
    client = OpenAI()
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = 0,
    )
    #return response.choices[0].message["content"]
    #return response['choices'][0]['message']['content']
    return response.choices[0].message.content
    

# print(get_completion("What is 1+1?"))

# customer_email = """
#                 Arrr, I be fuming that me blender lid \
#                 flew off and splattered me kitchen walls \
#                 with smoothie! And to make matters worse,\
#                 the warranty don't cover the cost of \
#                 cleaning up me kitchen. I need yer help \
#                 right now, matey!
#                 """
# style = """American English \
#         in a calm and respectful tone
#         """
# prompt = f"""Translate the text that is delimited by triple backitcks into a style that is {style}.
#         text: ```{customer_email}```
#         """
# response = get_completion(prompt)
# print(response)
#--------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------#
# LangChain 사용하기
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

chat = ChatOpenAI(temperature=0.0, model=llm_model)
pprint(chat)

template_string = """Translate the text that is delimited by triple backticks into \
a style that is {style}. text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)
print(prompt_template.messages[0].prompt.input_variables)
print(prompt_template.messages[0].prompt)

customer_email = """
                Arrr, I be fuming that me blender lid \
                flew off and splattered me kitchen walls \
                with smoothie! And to make matters worse,\
                the warranty don't cover the cost of \
                cleaning up me kitchen. I need yer help \
                right now, matey!
                """
customer_style = """American English in a calm and respectful tone"""
customer_messages = prompt_template.format_messages( 
    style=customer_style,
    text=customer_email,
)
'''
Q. format_messages, from_template 차이는??
- format_messages는 style, text를 입력받아서 prompt를 만들어준다.
- from_template은 template_string을 입력받아서 prompt를 만들어준다.
- format_messages는 입력 값이고, from_template은 입력 값이 아니라 template_string이다.
'''

# print(type(customer_messages))
# print(type(customer_messages[0]))
# print(customer_messages[0])
# # Call the LLM to translate to the style of the customer message
# customer_response = chat(customer_messages)
# print(customer_response.content)
# #--------------------------------------------------------------------------------#

service_reply = "Hey there customer, the warranty does not cover \
cleaning expenses for your kitchen because it's your fault that \
you misused your blender by forgetting to put the lid on before starting the blender. \
Tough luck! See ya!"

service_style_pirate = """a polite tone that speaks in English Pirate"""
service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)
#print(service_messages[0].content)
# service_response = chat(service_messages)
# print(service_response.content)
#--------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------#
# Output Parsers
# 1. format_instructions을 사용하지 않을 때 
# --> prompt_template.format_messages(text=customer_review)
"""
고객의 리뷰를 입력으로 받아서, gift, delivery_days, price_value를 출력하는 함수를 만들어본다.
출력 포맷은 JSON으로 한다.
"""
{
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(review_template) # 리뷰 템플릿을 정하고
#print(prompt_template)
print(prompt_template.messages[0].prompt)
messages = prompt_template.format_messages(text=customer_review) # 리뷰를 입력으로 넣는다.
# chat = ChatOpenAI(temperature=0.0, model=llm_model)
# response = chat(messages)
# print(response.content)
'''
{
    "gift": true,
    "delivery_days": 2,
    "price_value": ["It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features."]
}
'''
# print(type(response.content)) # 하지만 포맷은 딕셔너리는 아니다.
# print(response.content['price_value']) # 따라서, 다음과 같이 응답의 price_value만 출력할 수 없다.

#================================================================================================#
# format_instructions 구성
# from langchain.output_parsers import ResponseSchema
# from langchain.output_parsers import StructuredOutputParser
gift_schema = ResponseSchema(name="gift",
description="Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.")

delivery_days_schema = ResponseSchema(name="delivery_days",
description="How many days did it take for the product\
to arrive? If this information is not found, output -1.")

price_value_schema = ResponseSchema(name="price_value",
description="Extract any sentences about the value or \
price, and output them as a comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print(type(format_instructions))
print(format_instructions)
'''
```json
{
        "gift": string  // Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.
        "delivery_days": string  // How many days                                      did it take for the product                                      to arrive? If this                                       information is not found,                                      output -1.
        "price_value": string  // Extract any                                    sentences about the value or                                     price, and output them as a                                     comma separated Python list.
}
```
'''
#================================================================================================#
# 2. format_instructions을 사용할 때
# --> prompt_template.format_messages(text=customer_review, format_instructions=format_instructions)
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

# 앞의 예시와 차기아 있다면, format_instructions 유무이다.
# 
messages = prompt.format_messages(text=customer_review, 
                                  format_instructions=format_instructions) 
                                
print(messages[0].content)
response = chat(messages)
print(response.content) # 여전히 str 이다.
"""
```json
{
        "gift": true,
        "delivery_days": "2",
        "price_value": ["It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features."]
}
```
"""
output_dict = output_parser.parse(response.content) # response.content은 str이다.
print(type(output_dict))
print(output_dict)
print(output_dict.get('delivery_days'))
