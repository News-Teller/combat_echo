import numpy as np


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_similarities(row, target_emb):
    sample_emb = row.embedding

    return cosine_similarity(target_emb, sample_emb)


def get_most_similar(data, target, target_emb, num=5):
    similarities = []

    data.drop_duplicates(subset=["cleaned_important_text"], inplace=True)

    for row in data.iterrows():
        test_emb = row[1].embedding
        similarities.append(cosine_similarity(target_emb, test_emb))

    data["similarity"] = similarities

    copy = data.sort_values(by='similarity', ascending=False)

    copy = copy[copy["cleaned_important_text"] != target]

    copy.reset_index(drop=True, inplace=True)

    return copy.head(num)
