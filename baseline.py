import json
import torch
from rouge import Rouge
from bert_score import score as bert_score
from evaluate import load
from tqdm import tqdm

# Метрики
rouge_metric = Rouge()
bleurt_metric = load("bleurt", "bleurt-20")

# Устройство
device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_conclusion(text: str):
    """
    Извлекает текст после слова 'ЗАКЛЮЧЕНИЕ' (если найдено).
    """
    keyword = "заключение"
    lowered = text.lower()
    if keyword in lowered:
        index = lowered.find(keyword)
        print(text[index:].strip())
        return text[index:].strip()
    return None

def evaluate_metrics(reference, prediction):
    rouge = rouge_metric.get_scores(prediction, reference, avg=True)
    bleurt = bleurt_metric.compute(predictions=[prediction], references=[reference])["scores"][0]
    P, R, F1 = bert_score([prediction], [reference], lang="ru")
    return {
        "ROUGE-1": rouge["rouge-1"]["f"],
        "ROUGE-2": rouge["rouge-2"]["f"],
        "ROUGE-L": rouge["rouge-l"]["f"],
        "BLEURT": bleurt,
        "BERTScore": float(F1.mean())
    }

def evaluate_conclusions(json_path: str, max_samples=50):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for entry in tqdm(data[:max_samples]):
        full_text = entry.get("text", "")
        reference = entry.get("abstract", "")
        if not full_text.strip() or not reference.strip():
            continue

        conclusion = extract_conclusion(full_text)
        if conclusion:
            score = evaluate_metrics(reference, conclusion)
            results.append(score)

    # Подсчёт средних значений
    if results:
        from pandas import DataFrame
        df = DataFrame(results)
        print("\nСредние метрики по заключениям:")
        print(df.mean())
        df.to_excel("conclusion_baseline_metrics.xlsx", index=False)
    else:
        print("Ни в одном из примеров не найдено заключение.")

# Запуск
evaluate_conclusions("abstracts_texts.json", max_samples=150)
