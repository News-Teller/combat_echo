from news_diversification.src.preprocessing.preprocessing import preprocess_target
from news_diversification.src.result_ordering import divide_by_polarity_and_subjectivity
from news_diversification.src.similarity_calculation.similarity_calculation_fasttext import SimilarityFasttext
import pandas as pd

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

    output = divide_by_polarity_and_subjectivity(result, publication_date, random=False)

    for k, v in output.items():
        if len(v) == 2:
            print(f"{k} :\n {v[0]}\n {v[1]}")
        elif len(v) == 1:
            print(f"{k} :\n {v[0]}")
        else:
            print("No articles")

    print("OLD")

    print(result.iloc[0].url, result.iloc[0].similarities)
    print(result.iloc[1].url, result.iloc[1].similarities)
    print(result.iloc[2].url, result.iloc[2].similarities)
    print(result.iloc[3].url, result.iloc[3].similarities)
    print(result.iloc[4].url, result.iloc[4].similarities)


if __name__ == '__main__':
    url = "https://www.theguardian.com/uk-news/2021/apr/18/the-queen-alone-how-prince-philip-death-will-change-the-future-of-the-monarchy"
    main(url)
