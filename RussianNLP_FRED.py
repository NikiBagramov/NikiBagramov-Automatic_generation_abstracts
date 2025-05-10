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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import random
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

    cluster_to_sentences = {i: [] for i in range(k)}
    for i, label in enumerate(labels):
        dist = np.linalg.norm(emb[i] - kmeans.cluster_centers_[label])
        cluster_to_sentences[label].append((dist, i))

    for cluster_id in cluster_to_sentences:
        cluster_to_sentences[cluster_id].sort()

    selected_indices = []
    current_tokens = 0

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
                break
        if not added:
            break

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

from transformers import GPT2Tokenizer, T5ForConditionalGeneration

# --- Модель FRED-T5 ---
MODEL_NAME = "RussianNLP/FRED-T5-Summarizer"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, eos_token='</s>')
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

def generate_summary(input_text):
    prompt = f"<LM> Напиши аннотацию к научной статье.\n В данной статье рассматривается {input_text.strip()}"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

    output = model.generate(
        input_ids,
        max_length=750,
        min_length=120,
        num_beams=7,
        do_sample=True,
        temperature=0.65,
        top_p=0.9,
        no_repeat_ngram_size=4,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id
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

def test_fred_with_examples(json_path, k=3, max_samples=150):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)[:max_samples]

    all_results = []
    all_examples = []

    for name, ext_func in EXTRACTIVE.items():
        print(f"\n {name} → FRED-T5")
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

        avg = pd.DataFrame(metrics).mean().to_dict()
        avg.update({"Extractive": name, "Abstractive": "FRED-T5"})
        all_results.append(avg)
        print("Средние метрики:")
        for metric_name, value in avg.items():
            if metric_name not in ["Extractive", "Abstractive"]:
                print(f"  {metric_name}: {value:.4f}")

        print("\nПримеры генерации:")
        for ex in random.sample(local_examples, min(5, len(local_examples))):
            print(f"\n— {ex['Extractive Method']}")
            print(f"[Extracted]: {ex['Extracted Input'][:150]}...")
            print(f"[Generated]: {ex['Generated Summary']}")
            print(f"[Reference ]: {ex['Reference']}\n")

        all_examples.extend(local_examples)

    pd.DataFrame(all_results).to_excel("fred_summary_metrics.xlsx", index=False)
    pd.DataFrame(all_examples).to_excel("fred_summary_examples.xlsx", index=False)
    print("\nГотово! Сохранены файлы:")
    print("fred_summary_metrics.xlsx")
    print("fred_summary_examples.xlsx")


# Запуск
test_fred_with_examples("abstracts_texts.json", k=3, max_samples=22)
