from evaluation.fetch_data_for_evaluation import load_evaluation_data
from similarity_calculation.similarity_calculation_spacy import get_most_similar
import spacy
import pandas as pd
import time
NLP = spacy.load("en_core_web_lg")


def get_spacy_accuracy(df, url_mapping):
    accuracy = []

    for row in df.iterrows():
        url = row[1][0]
        target_clean = row[1][1]
        publication_date = row[1][3]
        url_emb = row[1][4]

        # print(url)
        # print(target_clean)

        expected_set = url_mapping[url]

        result = get_most_similar(df, target_clean, url_emb, num=len(expected_set))

        result_set = set(result.url)

        intersection = expected_set.intersection(result_set)

        acc = len(intersection) / len(expected_set)

        accuracy.append(acc)

        # print("Expected : ", expected_set)
        # print("Predicted : ", result_set)
        # print("Intersection : ", intersection)
        # print("Accuracy: ", acc)
        #
        # print("\n")

    total_acc = sum(accuracy) / len(accuracy)

    return total_acc


def get_spacy_embedding(tokens):
    tokens = NLP(tokens)
    return tokens.vector.tolist()


def preprocess_data_for_spacy(df):

    df["embedding"] = df["cleaned_important_text"].apply(get_spacy_embedding)

    df.to_csv("../../resources/evaluation/evaluation_cleaned_spacy.csv", index=False)
    return df


def main():
    df, test_urls, url_mapping = load_evaluation_data()

    # df = preprocess_data_for_spacy(df)

    df = pd.read_csv("../../resources/evaluation/evaluation_cleaned_spacy.csv")

    df["embedding"] = df.embedding.apply(eval)

    total_accuracy = get_spacy_accuracy(df, url_mapping)

    print(f"total accuracy {total_accuracy * 100} %")


if __name__ == '__main__':
    start = time.time()
    main()

    end = time.time()
    print(end - start)
