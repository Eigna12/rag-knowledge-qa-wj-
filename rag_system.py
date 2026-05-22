import os
import re
import pickle
import jieba
import math
from collections import Counter
import ollama

STOP_WORDS = set(['的', '了', '在', '是', '我', '有', '和', '就',
                  '不', '人', '都', '一', '一个', '上', '也', '很', '到',
                  '说', '要', '去', '你', '会', '着', '没有', '看', '好',
                  '自己', '这', '他', '她', '它', '们', '那', '些', '所'])


def tokenize(text):
    words = jieba.lcut(text)
    tokens = [w for w in words if w.strip() and w not in STOP_WORDS and len(w) > 1]
    # 额外保留数字和英文作为独立 token
    extra = re.findall(r'[\d\.]+|[a-zA-Z]+', text)
    tokens.extend(extra)
    return tokens


def load_documents(docs_dir="docs"):
    docs = []
    for file in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file)
        if file.endswith(('.md', '.txt')):
            content = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                print(f"警告：无法读取 {file}，跳过")
                continue
            docs.append({'source': file, 'content': content})
    print(f"已加载 {len(docs)} 个文档")
    return docs


def split_text(text, chunk_size=100):
    """更细粒度的切分，确保数值和说明不分离"""
    # 先按自然行切
    lines = text.split('\n')
    chunks = []
    current = ''
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(current + line) <= chunk_size:
            current += line + '\n'
        else:
            if current:
                chunks.append(current.strip())
            current = line + '\n'
    if current:
        chunks.append(current.strip())
    return chunks


def build_index(docs):
    all_chunks = []
    for doc in docs:
        chunks = split_text(doc['content'])
        for i, chunk in enumerate(chunks):
            tokens = tokenize(chunk)
            all_chunks.append({
                'source': doc['source'],
                'chunk_id': i,
                'content': chunk,
                'tokens': tokens
            })

    # 计算 TF-IDF 所需的数据
    N = len(all_chunks)
    # 词项到文档频率的映射
    df = Counter()
    for chunk in all_chunks:
        unique_tokens = set(chunk['tokens'])
        for token in unique_tokens:
            df[token] += 1

    # 构建每个 chunk 的 TF 向量（归一化）
    tf_vectors = []
    for chunk in all_chunks:
        freq = Counter(chunk['tokens'])
        total = len(chunk['tokens']) if chunk['tokens'] else 1
        tf = {token: count / total for token, count in freq.items()}
        tf_vectors.append(tf)

    index = {
        'chunks': all_chunks,
        'tf_vectors': tf_vectors,
        'df': df,
        'N': N
    }
    with open('data/index.pkl', 'wb') as f:
        pickle.dump(index, f)

    print(f"已构建索引，共 {len(all_chunks)} 个文本块")
    return index


def search(query, index, top_k=6):
    query = query.strip()
    if not query or len(query) < 2:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    # 计算查询的 TF-IDF 与每个文档的余弦相似度
    query_tf = Counter(query_tokens)
    query_norm = math.sqrt(sum(v * v for v in query_tf.values()))

    scores = []
    for i, tf_vec in enumerate(index['tf_vectors']):
        # 计算点积（TF-IDF）
        dot_product = 0
        doc_norm = 0
        for token, tf_idf_weight in tf_vec.items():
            # 计算该 token 的 IDF
            idf = math.log(index['N'] / (1 + index['df'].get(token, 0)))
            weight = tf_idf_weight * idf
            doc_norm += weight * weight
            if token in query_tf:
                dot_product += (query_tf[token] * weight)
        doc_norm = math.sqrt(doc_norm) if doc_norm > 0 else 0
        if doc_norm == 0 or query_norm == 0:
            score = 0
        else:
            score = dot_product / (query_norm * doc_norm)
        if score > 0:
            scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = scores[:top_k]

    results = []
    for idx, score in top_results:
        chunk = index['chunks'][idx]
        results.append({
            'source': chunk['source'],
            'score': score,
            'content': chunk['content']
        })
    return results


def generate_answer(query, retrieved_docs):
    context = "\n\n".join([doc['content'] for doc in retrieved_docs])

    prompt = f"""你是一个技术文档问答助手。请根据以下参考文档内容回答用户的问题。
如果文档中有相关的具体数据或改进结果，请务必完整引用，包括优化前后的数值。

参考文档：
{context}

用户问题：{query}

请用简洁清晰的中文回答："""

    response = ollama.chat(
        model='qwen2:1.5b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']


if __name__ == "__main__":
    print("=" * 50)
    print("RAG 智能问答系统 (TF-IDF + Qwen2)")
    print("=" * 50)

    docs = load_documents("docs")
    if not docs:
        print("docs/ 文件夹为空，请放入 .md 或 .txt 文件")
        exit(1)

    index = build_index(docs)

    print("\n💬 问答测试：输入问题，AI 将基于文档生成答案")
    while True:
        query = input("\n请输入问题（输入 q 退出）：")
        if query.lower() == 'q':
            break
        if not query.strip():
            continue

        retrieved = search(query, index)

        if not retrieved:
            print("未找到相关文档片段，无法回答该问题。")
            continue

        print(f"找到 {len(retrieved)} 个相关片段，正在生成答案...\n")
        answer = generate_answer(query, retrieved)
        print("=" * 50)
        print("📝 AI 回答：")
        print(answer)
        print("=" * 50)