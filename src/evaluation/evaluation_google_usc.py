import pandas as pd
import tensorflow_hub as hub
import numpy as np

from news_diversification.src.evaluation.fetch_data_for_evaluation import load_evaluation_data
from sklearn.metrics.pairwise import cosine_similarity


def get_google_usc_similarities(df, target_clean, target_embedding, num):
    embeddings = list(df.embedding)

    sims = cosine_similarity(np.array(target_embedding).reshape(1, -1), embeddings)

    pairs = []
    for i in range(sims.shape[1]):
        pairs.append({'index': i, 'score': sims[0][i]})

    temp = df

    temp["similarities"] = [pair["score"].item() for pair in pairs]

    temp = temp[temp["cleaned_important_text"] != target_clean]

    temp.sort_values(by='similarities', ascending=False, inplace=True)

    return temp.head(num)


def get_google_usc_accuracy(df, url_mapping):
    accuracy = []

    for row in df.iterrows():
        url = row[1][0]
        target_clean = row[1][1]
        publication_date = row[1][3]
        url_emb = row[1][4]

        print(url)
        print(target_clean)

        expected_set = url_mapping[url]

        result = get_google_usc_similarities(df, target_clean, url_emb, num=len(expected_set))

        result_set = set(result.url)

        intersection = expected_set.intersection(result_set)

        acc = len(intersection) / len(expected_set)

        accuracy.append(acc)

        print("Expected : ", expected_set)
        print("Predicted : ", result_set)
        print("Intersection : ", intersection)
        print("Accuracy: ", acc)

        print("\n")

    total_acc = sum(accuracy) / len(accuracy)

    return total_acc


def preprocess_data_for_google_usc(df):
    embed = hub.load("../../resources/universal-sentence-encoder_4")

    embeddings = embed(list(df.cleaned_important_text)).numpy().tolist()

    df["embedding"] = embeddings

    df.to_pickle("../../resources/evaluation/evaluation_cleaned_google_usc.pickle")
    return df


def main():
    df, test_urls, url_mapping = load_evaluation_data()

    df = preprocess_data_for_google_usc(df)

    df = pd.read_pickle("../../resources/evaluation/evaluation_cleaned_google_usc.pickle")

    total_accuracy = get_google_usc_accuracy(df, url_mapping)

    print(f"total accuracy {total_accuracy * 100} %")


if __name__ == '__main__':
    main()
