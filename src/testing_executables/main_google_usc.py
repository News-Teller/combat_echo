from preprocessing.preprocessing import preprocess_target_bert
import pandas as pd

from preprocessing.preprocessing_google_usc import GoogleUscPreprocessor
from similarity_calculation.similarity_calculation_google_usc import SimilarityGoogleUsc

CLEANED_DATA_PATH = "../../resources/cleaned_data.csv"

def main(url):
    target_clean, publication_date = preprocess_target_bert(url)  # TODO factor date

    print(target_clean)

    df = pd.read_csv(CLEANED_DATA_PATH)

    print("Preprocessing...")

    preprocessor = GoogleUscPreprocessor(df)

    preprocessor.calculate_embeddings_and_save()

    print("Done")

    calculator = SimilarityGoogleUsc()

    result = calculator.calculate_similarity_for_target(target_clean)

    print(result.url.iloc[0])
    print(result.url.iloc[1])
    print(result.url.iloc[2])
    print(result.url.iloc[3])
    print(result.url.iloc[4])


if __name__ == '__main__':
    url = "https://www.theguardian.com/uk-news/2021/apr/18/the-queen-alone-how-prince-philip-death-will-change-the-future-of-the-monarchy"

    main(url)
