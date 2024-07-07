from openai import OpenAI
import os
from flask import Flask, request, render_template, jsonify
import json
import jieba
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 從 JSON 文件中讀取 iframe 數據
def load_iframes():
    with open('iframes.json', 'r', encoding='utf-8') as f:
        return json.load(f)

iframes = load_iframes()

# 預處理 JSON 標題
def preprocess_titles():
    return [' '.join(jieba.cut(iframe['title'])) for iframe in iframes]

preprocessed_titles = preprocess_titles()

# 創建 TF-IDF 向量化器
vectorizer = TfidfVectorizer()
title_vectors = vectorizer.fit_transform(preprocessed_titles)

# 使用 ChatGPT 提取關鍵詞並生成相關替代詞
def extract_keywords_with_gpt(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一個能提取句子關鍵字並提供相關替代詞的助手。請使用繁體中文回答。"},
            {"role": "user", "content": f"請提取以下句子中的關鍵字並提供相關替代詞，用逗號分隔：{text}"}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    keywords = response.choices[0].message.content.strip().split("，")
    return keywords

# 查找與 JSON 數據最相近的關鍵詞
def find_best_matches(keywords, iframes, threshold=0.3):
    titles = [iframe['title'] for iframe in iframes]
    best_matches = []
    
    # 階段 1: 使用 get_close_matches
    for keyword in keywords:
        close_matches = get_close_matches(keyword, titles, n=3, cutoff=0.1)
        best_matches.extend(close_matches)
    
    # 階段 2: TF-IDF 相似度匹配
    query_vector = vectorizer.transform([' '.join(keywords)])
    similarities = cosine_similarity(query_vector, title_vectors).flatten()
    best_match_indices = np.where(similarities > threshold)[0]
    best_matches.extend([titles[i] for i in best_match_indices])
    
    best_matches = list(set(best_matches))
    return [iframe for iframe in iframes if iframe['title'] in best_matches]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    
    # 如果查詢以句號結尾，使用 GPT 生成回答
    if query.strip().endswith('。'):
        gpt_response = get_gpt_response(query)
        return jsonify({"response": gpt_response})
    
    # 使用 ChatGPT 提取關鍵詞並生成相關替代詞
    keywords = extract_keywords_with_gpt(query)
    
    # 查找相似的 iframes
    similar_iframes = find_best_matches(keywords, iframes)
    
    if similar_iframes:
        return jsonify(similar_iframes)
    else:
        # 如果沒有找到匹配，使用 GPT 生成回答
        gpt_response = get_gpt_response(query)
        return jsonify({"response": gpt_response})

# 使用 GPT 生成回答
def get_gpt_response(query):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一個友好的助手，專門回答有關城市生活、交通和公共服務的問題。請使用繁體中文回答，並盡量提供有用的信息或建議。"},
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in get_gpt_response: {e}")
        return "抱歉，我現在無法回答這個問題。請稍後再試。"

if __name__ == '__main__':
    app.run(debug=True)
