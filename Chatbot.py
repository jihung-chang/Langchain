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
    max_tokens=1000,
    openai_api_base=base_url if base_url else None
)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a prompt template and give the predefined system prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ".你是一個博學多聞的分子生物學教授，同時又能夠將專業的知識以通俗易懂的方式解釋給學生聽。\n"
            "你會根據學生的問題提供詳細的解釋和相關的背景知識。\n"
            "請注意，你的回答應該是專業的，但同時也要易於理解。\n"
            "如果學生的問題不夠清楚，你會請他們提供更多的細節。\n"
            "你會使用簡單的語言和例子來幫助學生理解。\n"
            "並用繁體中文回答。\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the function that calls the model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = chat_model.invoke(prompt)
    return {"messages": response}

# Add a node "model" into the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Create a config that pass into the runnable every time
config = {"configurable": {"thread_id": "abc123"}}

# 連續對話模式
print("=== 分子生物學教授聊天機器人 ===")
print("輸入 'quit'、'exit' 或 'bye' 來結束對話")
print("開始對話...")
print("-" * 50)

while True:
    try:
        # 從終端機獲取用戶輸入
        user_input = input("\n你: ").strip()
        
        # 檢查是否要結束對話
        if user_input.lower() in ['quit', 'exit', 'bye', '再見', '結束']:
            print("\n教授: 很高興與你討論分子生物學！再見！\n")
            break
            
        # 建立訊息並發送給模型
        input_messages = [HumanMessage(user_input)]
        output = app.invoke({"messages": input_messages}, config)
        
        # 顯示回應
        print(f"\n教授: {output['messages'][-1].content}")
        
    except Exception as e:
        print(f"\n發生錯誤: {e}")
        print("請重新輸入你的問題...")