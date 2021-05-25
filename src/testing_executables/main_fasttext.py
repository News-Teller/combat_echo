from preprocessing.preprocessing import preprocess_target
from similarity_calculation.similarity_calculation_fasttext import SimilarityFasttext
import pandas as pd

from live_processing.pca_diversification import get_most_diverse_articles

pd.set_option('display.max_columns', 500)


def main(url):
    # from_ = '2021-03-23T00:00:00.000'
    # to_ = '2021-03-23T23:59:00.000'
    #
    # df = fetch(from_, to_)
    # df = preprocess_cached(df)
    #
    # print(df.head())

    target_clean, publication_date = preprocess_target(url)

    # target_clean = clean_text("Hate crimes in Asia are not unique to the United States. An Asian man was brutally attacked in London during a live stream, but he received a great deal of help from an angry bystander.")

    # print(publication_date)

    print(target_clean)

    calculator = SimilarityFasttext()

    result = calculator.get_similarities(target_clean)

    result = get_most_diverse_articles(result, embedding_column="fasttext_embedding")

    # output = divide_by_polarity_and_subjectivity(result, publication_date, random=False)

    # for k, v in output.items():
    #     if len(v) == 2:
    #         print(f"{k} :\n {v[0]}\n {v[1]}")
    #     elif len(v) == 1:
    #         print(f"{k} :\n {v[0]}")
    #     else:
    #         print("No articles")

    print(result.url.iloc[0])
    print(result.url.iloc[1])
    print(result.url.iloc[2])

    print(result.important_text.loc[0])
    print(result.important_text.loc[1])
    print(result.important_text.loc[2])


if __name__ == '__main__':
    url = "https://www.independent.co.uk/news/world/americas/us-politics/qanon-conspiracy-biden-robot-cnn-b1809531.html"
    main(url)
