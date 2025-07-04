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

from langchain_openai import OpenAI

try:
    llm = OpenAI(
        model_name=model_name,
        api_key=api_key,
        temperature=0.7,
        max_tokens=100,
        openai_api_base=base_url if base_url else None
    )
except Exception as e:
    print(f"遇到錯誤: {str(e)}")


#########################################################################
# 示範 Agents, 你會得到一個 Warning 建議換使用 LangGraph 來 build agents
#########################################################################
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import tool

@tool
def calculate(expression: str) -> str:
    """進行簡單的數學計算"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"計算錯誤: {str(e)}"

try:
    # 初始化代理
    tools = load_tools(["llm-math"], llm=llm)
    tools.append(calculate)

    agent = initialize_agent(
        tools,
        chat_model,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # 使用代理解決問題
    agent_query = "如果我有5個蘋果，小明給了我3個，小華又拿走了2個，最後我總共有多少蘋果？"
    print(f"問題: {agent_query}")
    agent_result = agent.invoke({"input": agent_query})
    print(f"代理回答: {agent_result['output']}")
except Exception as e:
    print(f"代理示範遇到錯誤: {str(e)}")
print("----------------------------------------------------\n")


#########################################################################
# 示範 Chat History, memory的使用會有一個 Warning 建議換使用 LangGraph Memory
#########################################################################
# 7. 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

try:
    # 創建對話歷史和記憶
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        return_messages=True
    )

    # 創建一個帶歷史記錄的對話鏈
    template = """你是一個友好的助手。
    你需要保持對話的一致性和連貫性，記住之前的互動。

    當前對話:
    {chat_history}

    人類: {question}
    AI: """

    prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template,
    )

    chat_chain = prompt | chat_model | StrOutputParser()

    # 添加對話歷史功能
    chat_with_history = RunnableWithMessageHistory(
        chat_chain,
        lambda session_id: message_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # 進行多輪對話
    print("多輪對話示範:")

    # 第一輪
    response1 = chat_with_history.invoke(
        {"question": "你好，我叫小明。"},
        config={"configurable": {"session_id": "demo"}}
    )
    print("問: 你好，我叫小明。")
    print(f"答: {response1}")

    # 第二輪
    response2 = chat_with_history.invoke(
        {"question": "你還記得我叫什麼名字嗎？"},
        config={"configurable": {"session_id": "demo"}}
    )
    print("問: 你還記得我叫什麼名字嗎？")
    print(f"答: {response2}")

    print("\n對話歷史:")
    for message in message_history.messages:
        print(f"{message.type}: {message.content}")
except Exception as e:
    print(f"對話歷史示範遇到錯誤: {str(e)}")