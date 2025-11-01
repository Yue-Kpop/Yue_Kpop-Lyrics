import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from matplotlib.font_manager import FontProperties
import re
import ast # 引入 Abstract Syntax Tree 模組
from tqdm import tqdm

input_file = "processed_HYPE_data_ver2.csv"
df = pd.read_csv(input_file)
# ====================================================================
# ⚠️ 1. 資料讀取與準備 (請根據你的實際程式碼修改這部分)
# ====================================================================

# 假設你的 DataFrame 已經載入，並且已經完成了所有的預處理和合併步驟
# 例如： df = pd.read_csv('your_data.csv')
# 假設 df['final_tokens'] 欄位是 List of Strings 類型

# 確保 'final_tokens' 欄位中的每個字串都被安全地評估為 Python 列表
def convert_str_to_list(list_str):
    try:
        # ast.literal_eval 比 eval() 更安全，專門用於評估字串中的基本數據結構
        return ast.literal_eval(list_str)
    except (ValueError, TypeError):
        # 如果遇到 NaN 或無法評估的字串，返回空列表
        return []

# 應用轉換，這將是你的新最終詞彙欄位
df['final_tokens_restored'] = df['final_tokens'].apply(convert_str_to_list)

# --------------------------------------------------------------------
# 接下來的 LDA 流程，請使用這個新的還原欄位
# --------------------------------------------------------------------
# 1. 建立文檔集合 (List of Lists)
documents = df['final_tokens_restored'].tolist()

# 2. 移除空文檔（安全操作）
documents = [doc for doc in documents if doc]
# ====================================================================
# 2. 數據驗證與安全檢查 (避免 ValueError: cannot compute LDA over an empty collection)
# ====================================================================
total_docs = len(documents)
empty_docs_count = sum(1 for doc in documents if not doc)
total_tokens = sum(len(doc) for doc in documents)

print("\n--- 數據流失最終檢查 ---")
print(f"文檔總數 (歌曲數): {total_docs}")
print(f"空列表文檔數: {empty_docs_count}")
print(f"所有文檔中詞彙的總計數: {total_tokens}")

if total_tokens == 0:
    print("🚨 致命錯誤：所有文檔詞彙總計數為 0。請檢查 DataFrame 原始欄位。")
    exit()  # 停止執行

# --------------------------------------------------------------------
# 移除空文檔（如果空文檔數量不多，這樣可以避免它們干擾後續處理）
documents = [doc for doc in documents if doc]
# --------------------------------------------------------------------


# ====================================================================
# 3. 建立詞典 (Dictionary) 和語料庫 (Corpus)
# ====================================================================

print("\n開始建立詞典...")
# 使用所有文檔建立詞典
dictionary = corpora.Dictionary(documents)

# 詞彙過濾：使用最寬鬆的條件來避免丟失核心詞
dictionary.filter_extremes(
    no_below=2,  # 詞彙至少在 2 首歌中出現過
    no_above=0.99,  # 詞彙只有在超過 99% 的歌中出現才移除
    keep_n=None
)

print(f"✅ 詞彙表大小 (過濾後): {len(dictionary)}")

# 建立 BoW 語料庫 (將詞彙轉換為 (ID, Count) 格式)
corpus = [dictionary.doc2bow(doc) for doc in documents]
print(f"✅ 語料庫文檔數: {len(corpus)}")

# ====================================================================
# 4. 訓練 LDA 模型 (LdaModel)
# ====================================================================

# ⚠️ 關鍵參數： num_topics (建議從 10 開始嘗試)
NUM_TOPICS = 10

print(f"\n開始訓練 {NUM_TOPICS} 個主題的 LDA 模型...")

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,  # 設定隨機種子，確保結果可重現
    chunksize=100,
    passes=20,  # 增加迭代次數以提高模型品質
    alpha='auto',
    per_word_topics=False  # 這裡通常不需要 per_word_topics
)

print("✅ LDA 模型訓練完成。")

# ====================================================================
# 5. 結果解讀：檢視主題 (使用 CJK 字型確保韓文顯示)
# ====================================================================

# ⚠️ 請確保你的 FONT_PATH 指向一個支持韓文的字型，例如 Malgun Gothic
FONT_PATH = 'C:\\Windows\\Fonts\\malgun.ttf'
try:
    cjk_font = FontProperties(fname=FONT_PATH)
except:
    print("⚠️ 警告：無法載入韓文字型。終端機輸出可能會出現亂碼。")

print("\n--- LDA 主題模型結果 (Top 10 詞彙) ---")

for idx, topic in lda_model.print_topics(num_words=10):
    # 清理輸出格式：移除數字權重和小數點，只保留詞彙
    # 範例輸出: 0.050*"word" + 0.040*"word2"
    cleaned_topic = re.sub(r'\d\.\d{3}\*"', '', topic).replace('"', '').replace(' + ', ' / ')

    # 打印結果 (如果終端機支持，韓文會正常顯示)
    print(f"🌟 主題 #{idx + 1}：")
    print(f"   {cleaned_topic}\n")

# ====================================================================
# 6. 下一步：主題連貫性 (可選)
# ====================================================================

# 這是下一階段優化主題數量的關鍵
# from gensim.models.coherencemodel import CoherenceModel
# coherence_model_lda = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print(f"主題連貫性分數 (Coherence Score): {coherence_lda}")