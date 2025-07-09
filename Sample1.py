import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# 從環境變量獲取設定
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

#########################################################################
# 示範 LLM (用於文本生成), 因為公司的 ChatAI proxy 只支援 chat_completion endpoint，所以會出現 "Selected provider is forbidden" 錯誤訊息
#########################################################################
from langchain_openai import OpenAI

try:
    llm = OpenAI(
        model_name=model_name,
        api_key=api_key,
        temperature=0.7,
        max_tokens=100,
        openai_api_base=base_url if base_url else None
    )

    print("1. 示範 LLM Model (文本生成):")
    llm_result = llm.invoke("寫一首關於人工智能的短詩")
    print(f"示範 LLM Model 文本生成，寫一首關於人工智能的短詩: {llm_result}")
except Exception as e:
    print(f"遇到錯誤: {str(e)}")


#########################################################################
# 示範 Google Gemini LLM Model (用於文本生成)
#########################################################################
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        max_tokens=100,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print("\n2. 示範 Google Gemini LLM Model (文本生成):")
    gemini_result = gemini_model.invoke("寫一首關於人工智能的短詩，風格要輕鬆幽默")
    print(f"Gemini 模型生成的詩: {gemini_result.content}")
    
except Exception as e:
    print(f"\nGemini LLM 示範遇到錯誤: {str(e)}")


#########################################################################
# 示範 ChatModel (用於對話)
#########################################################################
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat_model = ChatOpenAI(
    model=model_name,
    temperature=0.5,
    max_tokens=256,
    openai_api_base=base_url if base_url else None
)

messages = [
    SystemMessage(content="你是一位非常善長溝通，講話婉轉但堅定的公關人員"),
    HumanMessage(content="公司業績太好，已經無法再接客戶訂單了，該如何回應客戶的詢問?")
]
chat_result = chat_model.invoke(messages)
print("\n3. 示範 ChatModel 對話生成")
print(f"專業公關的回覆: {chat_result.content}")
print(f"\nTotal token: {chat_result.response_metadata.get('token_usage').get('total_tokens')}")