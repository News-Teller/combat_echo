import numpy as np
from preprocessing import NLP


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_neighbours(data, target):
    similarities = list()
    target_emb = target.vector

    for row in data.iterrows():
        test_sample = row[1].cleaned_important_text
        sample_emb = test_sample.vector

        similarities.append(cosine_similarity(target_emb, sample_emb))

    data["similarities"] = similarities

    return data


def get_nearest_neighbours(data, target, num=5):
    neighbours = get_neighbours(data, target)

    return neighbours.nlargest(num, 'similarities')
