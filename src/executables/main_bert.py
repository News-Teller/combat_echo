# from result_ordering import divide_by_polarity_and_subjectivity
from news_diversification.src.preprocessing.preprocessing import preprocess_target_bert
from news_diversification.src.result_ordering import divide_by_polarity_and_subjectivity
from news_diversification.src.similarity_calculation.similarity_calculation_bert import SimilarityTransformer


def main(url):
    target_clean, publication_date = preprocess_target_bert(url)  # TODO factor date

    # target_clean = clean_text("Hate crimes in Asia are not unique to the United States. An Asian man was brutally attacked in London during a live stream, but he received a great deal of help from an angry bystander.")

    print(target_clean)

    transformer = SimilarityTransformer()

    result = transformer.calculate_similarity_for_target(target_clean)

    output = divide_by_polarity_and_subjectivity(result, publication_date, random=False)

    for k, v in output.items():
        if len(v) == 2:
            print(f"{k} :\n {v[0]}\n {v[1]}")
        else:
            print(f"{k} :\n {v[0]}")

    print("OLD")

    print(result.url.iloc[0])
    print(result.url.iloc[1])
    print(result.url.iloc[2])
    print(result.url.iloc[3])
    print(result.url.iloc[4])


if __name__ == '__main__':
    url = "https://www.theguardian.com/uk-news/2021/apr/18/the-queen-alone-how-prince-philip-death-will-change-the-future-of-the-monarchy"

    main(url)
