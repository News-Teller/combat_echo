from live_processing.media_filtering import perform_media_filtering, cache_media_dict
from preprocessing.preprocessing import preprocess_target
from news_diversification.src.similarity_calculation.similarity_calculation_google_usc import SimilarityGoogleUsc
from live_processing.pca_diversification import get_most_diverse_articles

import pandas as pd

CLEANED_DATA_PATH = "../../resources/cleaned_data.csv"

pd.set_option('display.max_columns', 500)


def main(url):
    target_clean, publication_date = preprocess_target(url)  # TODO factor date

    # target_clean = clean_text("Kamala Harris is the president and not Joe Biden")

    print(target_clean)

    calculator = SimilarityGoogleUsc()

    df = calculator.calculate_similarity_for_target(target_clean, threshold=0.1)

    filtered = perform_media_filtering(df)

    result = get_most_diverse_articles(filtered)

    print(result.url.loc[0])
    print(result.url.loc[1])
    print(result.url.loc[2])

    print(result.important_text.loc[0])
    print(result.important_text.loc[1])
    print(result.important_text.loc[2])

    print(result.bias.loc[0])
    print(result.bias.loc[1])
    print(result.bias.loc[2])

    print(result.fact.loc[0])
    print(result.fact.loc[1])
    print(result.fact.loc[2])


if __name__ == '__main__':
    # url = "https://www.theguardian.com/uk-news/2021/apr/18/the-queen-alone-how-prince-philip-death-will-change-the-future-of-the-monarchy"
    # url = "https://www.foxnews.com/opinion/biden-speech-top-takeaways-address-congress-david-bossie"
    # url = "https://thehill.com/blogs/congress-blog/foreign-policy/551313-the-afghanistan-withdrawal-that-could-have-been"
    url = "https://www.independent.co.uk/news/world/americas/us-politics/qanon-conspiracy-biden-robot-cnn-b1809531.html"

    main(url)
