from preprocessing.preprocessing import preprocess_target_bert, clean_text, \
    remove_spaces
from preprocessing.preprocessing_tfidf import TfidfPreprocessor
import pandas as pd

from similarity_calculation.similarity_calculation_tfidf import SimilarityTfidf

pd.set_option('display.max_columns', 500)


def main(url):
    df = pd.read_csv("../../resources/cleaned_data.csv")

    target, publication_date = preprocess_target_bert(url)
    target_clean = remove_spaces(clean_text(target))

    preprocessor = TfidfPreprocessor(df)
    preprocessor.run_pipeline()

    calculator = SimilarityTfidf()

    result = calculator.calculate_similarities_for_target(target, target_clean)

    print(result.iloc[0].url, result.iloc[0].similarities)
    print(result.iloc[1].url, result.iloc[1].similarities)
    print(result.iloc[2].url, result.iloc[2].similarities)
    print(result.iloc[3].url, result.iloc[3].similarities)
    print(result.iloc[4].url, result.iloc[4].similarities)


if __name__ == '__main__':
    url = "https://www.nytimes.com/live/2021/04/18/world/covid-vaccine-coronavirus-cases"
    main(url)