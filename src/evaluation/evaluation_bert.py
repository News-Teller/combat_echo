import pandas as pd
import time
from evaluation.fetch_data_for_evaluation import load_evaluation_data
from preprocessing.preprocessing_bert import BertPreprocessor
from similarity_calculation.similarity_calculation_bert import SimilarityTransformer


def get_bert_accuracy(df, url_mapping):
    accuracy = []

    transformer = SimilarityTransformer(data_path="../../resources/evaluation/evaluation_cleaned_bert.csv",
                                        embeddings_path="../../resources/evaluation/evaluation_embeddings.pt")
    for row in df.iterrows():
        url = row[1][0]
        target_clean = row[1][2]
        publication_date = row[1][3]

        # print(url)
        # print(target_clean)

        expected_set = url_mapping[url]

        result = transformer.calculate_similarity_for_target(target_clean, num=len(expected_set))

        result_set = set(result.url)

        intersection = expected_set.intersection(result_set)

        acc = len(intersection) / len(expected_set)

        accuracy.append(acc)

        # print("Expected : ", expected_set)
        # print("Predicted : ", result_set)
        # print("Intersection : ", intersection)
        # print("Accuracy: ", acc)

        # print("\n")

    total_acc = sum(accuracy) / len(accuracy)

    return total_acc


def preprocess_data_for_bert(df):
    preprocessorBert = BertPreprocessor(df, data_path="../../resources/evaluation/evaluation_cleaned_bert.csv",
                                        embeddings_path="../../resources/evaluation/evaluation_embeddings.pt")
    df = preprocessorBert.calculate_embeddings_and_save()

    return df


def main():
    df, test_urls, url_mapping = load_evaluation_data()

    # df = preprocess_data_for_bert(df)

    df = pd.read_csv("../../resources/evaluation/evaluation_cleaned_bert.csv")

    total_accuracy = get_bert_accuracy(df, url_mapping)

    print(f"total accuracy {total_accuracy * 100} %")


if __name__ == '__main__':
    start = time.time()
    main()

    end = time.time()
    print(end - start)
