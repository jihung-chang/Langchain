import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# 從環境變量獲取設定
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
api_key = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    model=model_name,
    temperature=0.5,
    max_tokens=256,
    openai_api_base=base_url if base_url else None
)


#########################################################################
# 示範 Prompt Template
#########################################################################
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 基本提示模板
basic_prompt = PromptTemplate.from_template(
    "請用一段話介紹{主題}，並列出三個與{主題}相關的重要概念。"
)
formatted_prompt = basic_prompt.format(主題="基因改造作物")
print(f"基本提示模板: {formatted_prompt}")
print ("--------------------------------------------------")


#########################################################################
# 示範 Chat Prompt Template, Output Parser, 以及 Chain 的執行 
#########################################################################
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# 字符串解析器
# str_parser = StrOutputParser()

# JSON 解析器
class Article(BaseModel):
    title: str = Field(description="文章的標題")
    points: List[str] = Field(description="文章的要點")

json_parser = JsonOutputParser(pydantic_object=Article)

# 聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """你需要提供一個關於指定主題的簡短文章。
     回應必須是一個有效的 JSON 格式，包含 'title' 和至少 3 個 'points'。"""),
    ("user", "請提供關於 {topic} 的內容")
])

# 創建和執行鏈
json_chain = chat_prompt | chat_model | json_parser
article = json_chain.invoke({"topic": "機器學習"})
print(f"JSON Parser 結果: {article}")