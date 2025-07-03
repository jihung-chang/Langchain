import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# 從環境變量獲取設定
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
api_key = os.getenv("OPENAI_API_KEY")


#########################################################################
# 示範 Embedding Models - 向量嵌入模型, 因為公司的 ChatAI proxy 只支援 chat_completion endpoint，所以會出現 "Selected provider is forbidden" 錯誤訊息
#########################################################################
from langchain_openai import OpenAIEmbeddings

# 初始化嵌入模型
try:
    embeddings = OpenAIEmbeddings(
        openai_api_base=base_url if base_url else None
    )

    # 示範文本嵌入
    text_samples = [
        "人工智能是計算機科學的一個分支",
        "機器學習是人工智能的核心技術",
        "深度學習是機器學習的一種方法"
    ]

    for text in text_samples:
        vector = embeddings.embed_query(text)
        print(f"文本: '{text}'\n向量維度: {len(vector)}\n向量片段: {vector[:3]}...\n")
except Exception as e:
    print(f"向量嵌入示範遇到錯誤: {str(e)}")
    print("-" * 30)
    

#########################################################################
# 示範 Vector Stores, 因為公司的 ChatAI proxy 只支援 chat_completion endpoint，所以會出現 "Selected provider is forbidden" 錯誤訊息
#########################################################################
from langchain_community.vectorstores import FAISS

# 建立一個簡單的文檔集合
documents = [
    "人工智能是研究如何使計算機模擬人類智能的科學",
    "機器學習是人工智能的一個子領域，專注於讓系統從數據中學習",
    "深度學習使用神經網絡結構來處理複雜的數據模式",
    "自然語言處理讓計算機理解和生成人類語言",
    "計算機視覺讓機器能夠理解和分析視覺信息"
]

try:
    # 創建向量存儲
    vectorstore = FAISS.from_texts(
        documents, 
        embeddings
    )

    # 進行相似性搜索
    query = "什麼是機器學習?"
    print(f"查詢: '{query}'")
    results = vectorstore.similarity_search(query, k=2)
    print("相似度搜索結果:")
    for doc in results:
        print(f"- {doc.page_content}\n")
except Exception as e:
    print(f"向量存儲示範遇到錯誤: {str(e)}")
