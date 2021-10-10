from src.preprocessing.preprocessing import preprocess_target_bert
import pandas as pd

from src.preprocessing.preprocessing_google_usc import GoogleUscPreprocessor
from src.preprocessing.preprocessing_scientific import preprocess_scientific_cached
from src.similarity_calculation.similarity_calculation_google_usc import SimilarityGoogleUsc

SCIENTIFIC_DATA_PATH = "../../resources/external/metadata.csv"
CLEANED_SCIENTIFIC_DATA_PATH = "../../resources/external/cleaned_scientific_data.csv"

SCIENTIFIC_GOOGLE_EMBEDDINGS = "../../resources/scientific_cleaned_google_usc.pickle"

def main(url):

    target_clean, publication_date = preprocess_target_bert(url)  # TODO factor date

    print(target_clean)

    calculator = SimilarityGoogleUsc(data_path=SCIENTIFIC_GOOGLE_EMBEDDINGS)

    result = calculator.calculate_similarity_for_target(target_clean)

    print(result.iloc[0])

    #
    #
    # print(result.url.iloc[0], result.important_text.iloc[0])
    # print(result.url.iloc[1], result.important_text.iloc[1])
    # print(result.url.iloc[2], result.important_text.iloc[2])
    # print(result.url.iloc[3], result.important_text.iloc[3])
    # print(result.url.iloc[4], result.important_text.iloc[4])


if __name__ == '__main__':
    url = "https://www.foxnews.com/health/covid-vaccine-prevention-cases-death-seniors-hhs"

    main(url)
