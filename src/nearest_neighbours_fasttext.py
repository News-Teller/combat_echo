import fasttext
import numpy as np
import pandas as pd


def train_model(train, save_model=True):
    cached = pd.DataFrame(train["cleaned_important_text"])
    cached.reset_index(drop=True, inplace=True)
    np.savetxt(r'../resources/train.txt', cached.values, fmt="%s")

    print("Training model...")
    model = fasttext.train_unsupervised("../resources/train.txt", )
    print("Done")

    if save_model:
        model.save_model("../resources/model.bin")


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_neighbours(model, data, target):
    similarities = list()

    for row1 in data.iterrows():
        test_sample = row1[1].cleaned_important_text

        emb1 = model.get_word_vector(target)
        emb2 = model.get_word_vector(test_sample)
        similarities.append(cosine_similarity(emb1, emb2))

    data["similarities"] = similarities

    return data


def get_nns(data, target):
    model = fasttext.load_model("../resources/model.bin")

    neighbours = get_neighbours(model, data, target)

    return neighbours.nlargest(5, 'similarities')
