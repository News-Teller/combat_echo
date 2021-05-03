import pandas as pd

from news_diversification.src.evaluation.fetch_data_for_evaluation import load_evaluation_data
from news_diversification.src.preprocessing.preprocessing import preprocess_target
from news_diversification.src.preprocessing.preprocessing_fasttext import FasttextPreprocessor
from news_diversification.src.similarity_calculation.similarity_calculation_fasttext import SimilarityFasttext


def get_fasttext_accuracy(df, url_mapping):
    accuracy = []

    transformer = SimilarityFasttext(
        cleaned_data_path="../../resources/evaluation/evaluation_cleaned_data_fasttext.pickle",
        model_path="../../resources/evaluation/evaluation_model.bin")
    for row in df.iterrows():
        url = row[1][0]
        target_clean = row[1][1]
        publication_date = row[1][3]

        print(url)
        print(target_clean)

        expected_set = url_mapping[url]

        result = transformer.get_similarities(target_clean, num=len(expected_set))

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


def preprocess_data_for_fasttext(df):

    preprocessorFasttext = FasttextPreprocessor(df, training_path="../../resources/evaluation/evaluation_train.txt",
                                                cleaned_data_path="../../resources/evaluation/evaluation_cleaned_data_fasttext.pickle",
                                                model_path="../../resources/evaluation/evaluation_model.bin")
    df = preprocessorFasttext.run_pipeline()

    return df


def main():
    df, test_urls, url_mapping = load_evaluation_data()

    # df = preprocess_data_for_fasttext(df)

    df = pd.read_pickle("../../resources/evaluation/evaluation_cleaned_data_fasttext.pickle")

    total_accuracy = get_fasttext_accuracy(df, url_mapping)

    print(f"total accuracy {total_accuracy * 100} %")


if __name__ == '__main__':
    main()
