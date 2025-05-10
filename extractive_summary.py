import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from rouge import Rouge
from bert_score import score as bert_score
from evaluate import load
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import random
import nltk
nltk.download('punkt')

# Настройки
bleurt_metric = load("bleurt", "bleurt-20")
rouge_metric = Rouge()
sbert_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# --- Экстрактивные методы ---
def sbert_centroid(text, k=6):
    sentences = sent_tokenize(text)
    if len(sentences) <= k:
        return text
    emb = sbert_model.encode(sentences)
    centroid = np.mean(emb, axis=0)
    scores = cosine_similarity(emb, centroid.reshape(1, -1)).flatten()
    top_idx = np.argsort(scores)[-k:]
    return " ".join([sentences[i] for i in sorted(top_idx)])

def sbert_pagerank(text, k=6):
    sentences = sent_tokenize(text)
    if len(sentences) <= k: return text
    emb = sbert_model.encode(sentences)
    sim_matrix = cosine_similarity(emb)
    scores = np.ones(len(sentences))
    for _ in range(10):
        scores = 0.85 * sim_matrix.dot(scores) + 0.15
    top_idx = np.argsort(scores)[-k:]
    return " ".join([sentences[i] for i in sorted(top_idx)])

def sbert_kmeans(text, k=6):
    sentences = sent_tokenize(text)
    if len(sentences) <= k:
        return text

    emb = sbert_model.encode(sentences)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(emb)
    labels = kmeans.labels_

    cluster_to_sentences = {i: [] for i in range(k)}
    for i, label in enumerate(labels):
        dist = np.linalg.norm(emb[i] - kmeans.cluster_centers_[label])
        cluster_to_sentences[label].append((dist, i))

    for cluster_id in cluster_to_sentences:
        cluster_to_sentences[cluster_id].sort()

    selected_indices = []
    for cluster_id in sorted(cluster_to_sentences.keys()):
        if cluster_to_sentences[cluster_id]:
            selected_indices.append(cluster_to_sentences[cluster_id][0][1])

    selected_indices = sorted(selected_indices)
    return " ".join([sentences[i] for i in selected_indices])

def textrank(text, k=6):
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
    "SBERT_KMeans": sbert_kmeans,
    "SBERT_Centroid": sbert_centroid,
    "SBERT_PageRank": sbert_pagerank,
    "TextRank": textrank
}

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

# --- Основной запуск ---
def test_extractive_only(json_path, k=6, max_samples=30):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)[:max_samples]

    all_results = []
    all_examples = []

    for name, ext_func in EXTRACTIVE.items():
        print(f"\n{name}")
        metrics = []
        local_examples = []

        for entry in tqdm(data):
            full, ref = entry["text"], entry["abstract"]
            if not full.strip() or not ref.strip():
                continue
            extract = ext_func(full, k)
            score = evaluate(ref, extract)
            metrics.append(score)
            local_examples.append({
                "Method": name,
                "Extracted Summary": extract[:350],
                "Reference": ref,
                "BLEURT": score["BLEURT"],
                "BERTScore": score["BERTScore"],
                "ROUGE-1": score["ROUGE-1"],
                "ROUGE-2": score["ROUGE-2"],
                "ROUGE-L": score["ROUGE-L"]
            })

        avg = pd.DataFrame(metrics).mean().to_dict()
        avg.update({"Method": name})
        all_results.append(avg)
        print("Средние метрики:")
        for metric_name, value in avg.items():
            if metric_name != "Method":
                print(f"  {metric_name}: {value:.4f}")

        print("\nПримеры:")
        for ex in random.sample(local_examples, min(5, len(local_examples))):
            print(f"\n— {ex['Method']}")
            print(f"[Extracted]: {ex['Extracted Summary']}")
            print(f"[Reference ]: {ex['Reference']}\n")

        all_examples.extend(local_examples)

    pd.DataFrame(all_results).to_excel("extractive_summary_metrics.xlsx", index=False)
    pd.DataFrame(all_examples).to_excel("extractive_summary_examples.xlsx", index=False)
    print("\nГотово! Сохранены файлы:")
    print("extractive_summary_metrics.xlsx")
    print("extractive_summary_examples.xlsx")

# Запуск
test_extractive_only("abstracts_texts.json", k=6, max_samples=150)
