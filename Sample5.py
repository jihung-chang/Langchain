import os
from dotenv import load_dotenv

load_dotenv(override=True)


##############################################################################################
# 示範 RAG (Retrieval-Augmented Generation), 使用 PDF 文檔作為知識庫，進行問答
# 使用到 Document Loaders, Text Splitters, Vector Stores, Embeddings, LLMs 和 Retrieval Chains
##############################################################################################
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

def create_rag_chain(pdf_path):
    """
    創建 RAG 問答鏈
    
    Args:
        pdf_path: PDF文件路徑
        
    Returns:
        qa_chain: 問答鏈對象
    """
    try:
        # 從環境變數獲取設定
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        
        print(f"正在載入PDF文件: {pdf_path}")
        
        # 檢查文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"找不到PDF文件: {pdf_path}")
        
        # Parsing Sales_AI_Agent.pdf 時因PDF格式問題會產生大量警告信息，暫時轉導 stdout 和 stderr 乎略這些 warnings
        import sys
        from io import StringIO
        
        # 保存原始 stdout 和 stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # 重定向输出到字符串緩沖區
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            # 1. 使用 PDF document loader 載入 PDF 文件
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
        finally:
            # 恢復原始输出
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        print(f"成功載入文件，共 {len(documents)} 頁")

        # 2. 將文件切割成較小的段落（chunk），最多 1000 個字符，重疊 100 個字符以銜接上下文
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        print(f"文件切割完成，共 {len(docs)} 個片段")

        # 3. 建立向量嵌入 (embedding)
        embeddings = OpenAIEmbeddings(
            openai_api_base=base_url if base_url else None,
            openai_api_key=api_key
        )

        # 4. 建立向量資料庫 (此處以 FAISS 為例)
        print("正在建立向量資料庫...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        print("向量資料庫建立完成")

        # 5. 建立檢索器 Retriever：返回最相似的 3 個文檔片段
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # 6. 建立基於檢索的問答鏈
        llm = ChatOpenAI(
            model=model_name, 
            temperature=0.3,
            max_tokens=500,
            openai_api_base=base_url if base_url else None,
            openai_api_key=api_key
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True  # 返回參考文檔
        )
        
        print("RAG 問答鏈建立完成！")
        return qa_chain
        
    except FileNotFoundError as e:
        print(f"文件錯誤: {e}")
        raise
    except Exception as e:
        print(f"建立 RAG 鏈時遇到錯誤: {str(e)}")
        if "Selected provider is forbidden" in str(e):
            print("提示: 這可能是因為公司的 ChatAI proxy 限制了 OpenAI Embeddings API 的訪問")
        raise


if __name__ == "__main__":
    try:
        pdf_file = "Sales_AI_Agent.pdf"
        rag_chain = create_rag_chain(pdf_file)
        
        # 測試問題清單
        queries = [
            "這份文件的主要內容是什麼？",
            "Sales AI Agent 有什麼功能？",
            "文件中提到了哪些關鍵技術？"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n問題 {i}: {query}")
            print("-" * 40)
            
            # 使用 invoke 方法而不是已棄用的 run 方法
            result = rag_chain.invoke({"query": query})
            
            print(f"回答: {result['result']}")
            
            # 顯示參考的源文檔
            if result.get('source_documents'):
                print(f"\n參考文檔片段:")
                for j, doc in enumerate(result['source_documents'], 1):
                    # 顯示文檔的前100個字符
                    content_preview = doc.page_content[:100].replace('\n', ' ')
                    page_num = doc.metadata.get('page', 'N/A')
                    print(f"  {j}. 頁面 {page_num}: {content_preview}...")
            
            print("\n" + "=" * 60)
            
    except Exception as e:
        print(f"執行過程中遇到錯誤: {str(e)}")