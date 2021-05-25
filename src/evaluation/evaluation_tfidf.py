import pandas as pd
import time
from evaluation.fetch_data_for_evaluation import load_evaluation_data
from preprocessing.preprocessing_tfidf import TfidfPreprocessor
from similarity_calculation.similarity_calculation_tfidf import SimilarityTfidf

pd.set_option('display.max_columns', 500)


def get_tfidf_accuracy(df, url_mapping):
    accuracy = []

    calculator = SimilarityTfidf(
        cleaned_data_location="../../resources/evaluation/evaluation_cleaned_tfidf.csv",
        ner_location="../../resources/evaluation/evaluation_extracted_named_entities.pickle",
        entities_weight_location="../../resources/evaluation/evaluation_extracted_named_entities_weights.pickle",
        tfidf_vectorizer_location="../../resources/evaluation/evaluation_tfidf_vectorizer.pk",
        tfidf_matrix_location="../../resources/evaluation/evaluation_tfidf_matrix.pk")

    for row in df.iterrows():
        url = row[1][0]
        target_clean = row[1][1]
        target = row[1][2]
        publication_date = row[1][3]

        # print(url)
        # print(target_clean)
        # print(target)

        expected_set = url_mapping[url]

        result = calculator.calculate_similarities_for_target(target, target_clean, num=len(expected_set), ner=True)

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


def preprocess_data_for_tfidf(df):
    preprocessorTfidf = TfidfPreprocessor(df,
                                          ner_location="../../resources/evaluation/evaluation_extracted_named_entities.pickle",
                                          entities_weight_location="../../resources/evaluation/evaluation_extracted_named_entities_weights.pickle",
                                          tfidf_vectorizer_location="../../resources/evaluation/evaluation_tfidf_vectorizer.pk",
                                          tfidf_matrix_location="../../resources/evaluation/evaluation_tfidf_matrix.pk")
    preprocessorTfidf.run_pipeline()

    df.to_csv("../../resources/evaluation/evaluation_cleaned_tfidf.csv", index=False)

    return df


def main():
    df, test_urls, url_mapping = load_evaluation_data()

    # df = preprocess_data_for_tfidf(df)

    print("Starting...")

    df = pd.read_csv("../../resources/evaluation/evaluation_cleaned_tfidf.csv")

    total_accuracy = get_tfidf_accuracy(df, url_mapping)

    print(f"total accuracy {total_accuracy * 100} %")


if __name__ == '__main__':

    start = time.time()
    main()

    end = time.time()
    print(end - start)
