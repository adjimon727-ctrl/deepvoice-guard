from collections.abc import Iterable, Mapping


def normalize_classification_scores(
    predictions: Iterable[Mapping[str, object]],
) -> dict[str, float]:
    scores = {"fake": 0.0, "real": 0.0}
    fake_labels = {"fake", "deepfake", "ai", "artificial", "synthetic", "spoof"}
    real_labels = {"real", "human", "authentic", "genuine"}
    recognized_label = False

    for prediction in predictions:
        label = prediction.get("label")
        score = prediction.get("score")
        if not isinstance(label, str) or not isinstance(score, (int, float)):
            continue

        normalized_label = label.casefold().replace("_", " ").replace("-", " ")
        label_words = set(normalized_label.split())
        if label_words & fake_labels:
            scores["fake"] = round(float(score) * 100, 2)
            recognized_label = True
        elif label_words & real_labels:
            scores["real"] = round(float(score) * 100, 2)
            recognized_label = True

    if not recognized_label:
        raise ValueError("Le modèle a renvoyé des classes de prédiction non reconnues.")

    return scores
