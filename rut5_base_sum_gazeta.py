#Отсюда
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from rouge import Rouge
from bert_score import score as bert_score
from evaluate import load
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import random

from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

# Настройки
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bleurt_metric = load("bleurt", "bleurt-20")
rouge_metric = Rouge()
sbert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# --- Экстрактивные методы ---
def full_pass(text, k=3):
    return text

def sbert_centroid(text, k=3):
    k = int(k)  # Приведение к целому числу
    sentences = sent_tokenize(text)
    if len(sentences) <= k:
        return text
    emb = sbert_model.encode(sentences)
    centroid = np.mean(emb, axis=0)
    scores = cosine_similarity(emb, centroid.reshape(1, -1)).flatten()
    top_idx = np.argsort(scores)[-k:]
    return " ".join([sentences[i] for i in sorted(top_idx)])


def sbert_pagerank(text, k=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= k: return text
    emb = sbert_model.encode(sentences)
    sim_matrix = cosine_similarity(emb)
    scores = np.ones(len(sentences))
    for _ in range(10):
        scores = 0.85 * sim_matrix.dot(scores) + 0.15
    top_idx = np.argsort(scores)[-k:]
    return " ".join([sentences[i] for i in sorted(top_idx)])

def sbert_kmeans(text, k=5, max_tokens=512):
    sentences = sent_tokenize(text)
    if len(sentences) <= k:
        return text

    emb = sbert_model.encode(sentences)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(emb)
    labels = kmeans.labels_

    # Кластер → список (дистанция, индекс)
    cluster_to_sentences = {i: [] for i in range(k)}
    for i, label in enumerate(labels):
        dist = np.linalg.norm(emb[i] - kmeans.cluster_centers_[label])
        cluster_to_sentences[label].append((dist, i))

    # Сортируем по близости к центру, а потом по порядку предложений
    for cluster_id in cluster_to_sentences:
        cluster_to_sentences[cluster_id].sort()

    selected_indices = []
    current_tokens = 0

    # Проходим по кластерам по очереди и жадно добавляем предложения
    while True:
        added = False
        for cluster_id in sorted(cluster_to_sentences.keys()):
            for _, idx in cluster_to_sentences[cluster_id]:
                if idx in selected_indices:
                    continue
                sent = sentences[idx]
                token_count = len(tokenizer.encode(sent, add_special_tokens=False))
                if current_tokens + token_count > max_tokens:
                    continue
                selected_indices.append(idx)
                current_tokens += token_count
                added = True
                break  # Только одно предложение за проход из кластера
        if not added:
            break

    # Сортируем по исходному порядку (для сохранения связности)
    selected_indices = sorted(selected_indices)
    return " ".join([sentences[i] for i in selected_indices])


def textrank_real(text, k=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= k: return text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)
    ranked = sorted(((score, idx) for idx, score in scores.items()), reverse=True)
    top_indices = sorted([idx for _, idx in ranked[:k]])
    return " ".join([sentences[i] for i in top_indices])

EXTRACTIVE = {
    "FullText": full_pass,
    "SBERT_KMeans": sbert_kmeans,
    "SBERT_Centroid": sbert_centroid,
    "SBERT_PageRank": sbert_pagerank,
    "TextRank": textrank_real
}

# --- Модель ruT5 ---
MODEL_NAME = "IlyaGusev/rut5_base_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def generate_summary(input_text):
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )["input_ids"].to(device)

    output = model.generate(
        input_ids,
        max_new_tokens=350,
        num_beams=7,
        no_repeat_ngram_size=4,
        early_stopping=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- Метрики ---
def evaluate(gold, pred):
    rouge = rouge_metric.get_scores(pred, gold, avg=True)
    bleurt = bleurt_metric.compute(predictions=[pred], references=[gold])["scores"][0]
    P, R, F1 = bert_score([pred], [gold], lang="ru")
    return {
        "ROUGE-1": rouge["rouge-1"]["f"],
        "ROUGE-2": rouge["rouge-2"]["f"],
        "ROUGE-L": rouge["rouge-l"]["f"],
        "BLEURT": bleurt,
        "BERTScore": float(F1.mean())
    }

def test_rut5_with_examples(json_path, k=3, max_samples=30):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)[:max_samples]

    all_results = []
    all_examples = []

    for name, ext_func in EXTRACTIVE.items():
        print(f"\n🔍 {name} → ruT5")
        metrics = []
        local_examples = []

        for entry in tqdm(data):
            full, ref = entry["text"], entry["abstract"]
            if not full.strip() or not ref.strip():
                continue
            extract = ext_func(full, k)
            summary = generate_summary(extract)
            score = evaluate(ref, summary)
            metrics.append(score)
            local_examples.append({
                "Extractive Method": name,
                "Extracted Input": extract[:350],
                "Generated Summary": summary,
                "Reference": ref,
                "BLEURT": score["BLEURT"],
                "BERTScore": score["BERTScore"],
                "ROUGE-1": score["ROUGE-1"],
                "ROUGE-2": score["ROUGE-2"],
                "ROUGE-L": score["ROUGE-L"]
            })

        # Средние метрики
        avg = pd.DataFrame(metrics).mean().to_dict()
        avg.update({"Extractive": name, "Abstractive": "ruT5"})
        all_results.append(avg)
        print("Средние метрики:")
        for metric_name, value in avg.items():
            if metric_name not in ["Extractive", "Abstractive"]:
                print(f"  {metric_name}: {value:.4f}")

        # 5 случайных примеров
        print("\nПримеры генерации:")
        for ex in random.sample(local_examples, min(5, len(local_examples))):
            print(f"\n— {ex['Extractive Method']}")
            print(f"[Extracted]: {ex['Extracted Input'][:150]}...")
            print(f"[Generated]: {ex['Generated Summary']}")
            print(f"[Reference ]: {ex['Reference']}\n")

        all_examples.extend(local_examples)

    # Сохранение
    pd.DataFrame(all_results).to_excel("rut5_summary_metrics.xlsx", index=False)
    pd.DataFrame(all_examples).to_excel("rut5_summary_examples.xlsx", index=False)
    print("\nГотово! Сохранены файлы:")
    print("rut5_summary_metrics.xlsx")
    print("rut5_summary_examples.xlsx")


# Запуск
test_rut5_with_examples("abstracts_texts.json", k=3, max_samples=30)
