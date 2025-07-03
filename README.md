# LangChain 示範專案

這是一個 LangChain 學習和示範專案，展示了如何使用 LangChain 框架進行各種 AI 應用開發，包括 RAG (Retrieval-Augmented Generation) 系統、聊天機器人和文檔問答系統。

## 📁 專案結構

```
📦 LangChain/
├── 📄 Sample1.py          # LLM 和 ChatModel 基礎示範 (含 OpenAI 和 Google Gemini)
├── 📄 Sample2.py          # Prompt Templates 和 Output Parsers 示範
├── 📄 Sample3.py          # Embedding Models 和 Vector Stores 示範
├── 📄 Sample4.py          # Agents 和 Tools 示範
├── 📄 Sample5.py          # RAG 系統示範
├── 📄 Sales_AI_Agent.pdf  # RAG 示範用的測試文檔
├── 📄 requirements.txt    # Python 套件相依性列表
├── 📄 .env.example        # 環境變數設定範例
├── 📄 .env                # 環境變數設定檔 (不會被 git 追蹤)
└── 📄 .gitignore          # Git 忽略檔案設定
```

## 🚀 快速開始

### 1. 環境準備

確保您的系統已安裝 Python 3.8 或更高版本：

```bash
python --version
```

### 2. 安裝相依套件

使用 pip 安裝所有必要的套件：

```bash
pip install -r requirements.txt
```

### 3. 環境變數設定

複製環境變數範例檔案並設定您的 API 金鑰：

```bash
cp .env.example .env
```

編輯 `.env` 檔案，填入您的 API 金鑰：

```env
# OpenAI API 設定 (透過公司 ChatAI proxy)
OPENAI_BASE_URL=https://openai-proxy-apigw-genai.api.linecorp.com/v1
OPENAI_MODEL=gpt-4o
OPENAI_API_KEY=<您的 ChatAI proxy token>

# Google Gemini API 金鑰 (需要到 https://aistudio.google.com/app/apikey 獲取)
GOOGLE_API_KEY=<您的 Google API 金鑰>
```

### 4. 執行示範程式

```bash
# 執行 LLM 基礎示範
python Sample1.py

# 執行 RAG 系統示範
python Sample5.py

# 執行其他功能示範
python Sample2.py  # 提示工程
python Sample3.py  # 向量處理
python Sample4.py  # 智能代理
```

## 📚 檔案說明

### 核心示範檔案

| 檔案 | 功能描述 | 主要展示內容 |
|------|----------|-------------|
| `Sample1.py` | LLM 和 ChatModel 基礎功能 | OpenAI GPT、Google Gemini 模型調用 |
| `Sample2.py` | 提示工程和輸出解析 | PromptTemplate、OutputParser 使用 |
| `Sample3.py` | 向量嵌入和向量資料庫 | Embeddings、FAISS 向量搜尋 |
| `Sample4.py` | 智能代理和工具整合 | Agents、Tools 的組合使用 |
| `Sample5.py` | RAG 系統 | PDF 文檔載入、向量化、問答 |

### 配置檔案

- **`.env`**: 環境變數設定，包含 API 金鑰等敏感資訊
- **`.env.example`**: 環境變數設定範例，可安全地分享
- **`.gitignore`**: Git 版本控制忽略檔案，保護敏感資訊
- **`requirements.txt`**: Python 套件相依性列表

## 🔧 主要功能

### 1. LLM 整合
- ✅ OpenAI GPT 模型 (透過公司 proxy)
- ✅ Google Gemini 模型
- ✅ 對話記憶管理
- ✅ 溫度和參數調整

### 2. RAG (檢索增強生成)
- ✅ PDF 文檔載入和解析
- ✅ 文本分割和向量化
- ✅ FAISS 向量資料庫
- ✅ 相似性搜尋和檢索
- ✅ 上下文感知問答

### 3. 提示工程
- ✅ 動態提示模板
- ✅ 結構化輸出解析
- ✅ 多語言支持

### 4. 向量處理
- ✅ 文檔嵌入
- ✅ 向量資料庫管理
- ✅ 語義搜尋

## 🔑 API 金鑰設定

### OpenAI API (透過公司 ChatAI proxy)
1. 聯繫您的 IT 部門獲取 ChatAI proxy token
2. 在 `.env` 檔案中設定 `OPENAI_API_KEY`

### Google Gemini API
1. 前往 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 建立新的 API 金鑰
3. 在 `.env` 檔案中設定 `GOOGLE_API_KEY`

## ⚠️ 注意事項

### 網絡限制
- 公司網絡可能限制某些 API 的訪問
- 如遇到 "Selected provider is forbidden" 錯誤，請：
  1. 檢查網絡連接
  2. 確認 API 金鑰正確
  3. 聯繫 IT 部門開放相關 API 訪問權限

### 檔案大小
- PDF 檔案建議小於 10MB
- 大型文檔會影響處理速度

### 成本控制
- API 調用會產生費用，請適度使用
- 建議設定合理的 `max_tokens` 限制

## 🛠️ 故障排除

### 常見問題

1. **模組導入錯誤**
   ```bash
   pip install -r requirements.txt
   ```

2. **API 金鑰錯誤**
   - 檢查 `.env` 檔案設定
   - 確認金鑰格式正確

3. **PDF 解析警告**
   - 程式已自動處理 PyPDF 警告
   - 不影響功能正常運作

4. **向量資料庫創建失敗**
   - 通常是網絡限制導致
   - 嘗試使用個人網絡或聯繫 IT 支援

## 📝 版本記錄

- **v1.0.0** - 初始版本，包含基礎 LangChain 功能示範
- **v1.1.0** - 新增 Google Gemini 支援和 RAG 系統
- **v1.2.0** - 改善錯誤處理和新增繁體中文支援

## 📄 授權

此專案僅供學習和研究使用。