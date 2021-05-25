from news_diversification.src.similarity_calculation.similarity_calculation_spacy import get_most_similar
from preprocessing.preprocessing import preprocess_target
from preprocessing.preprocessing import get_embedding

import pandas as pd

from live_processing.pca_diversification import get_most_diverse_articles

CLEANED_DATA_PATH = "../../resources/cleaned_data.csv"


def main(url):
    target_clean, publication_date = preprocess_target(url)  # TODO factor date

    # target_clean = clean_text("Hate crimes in Asia are not unique to the United States. An Asian man was brutally attacked in London during a live stream, but he received a great deal of help from an angry bystander.")

    print(target_clean)

    url_emb = get_embedding(target_clean)
    corpus = pd.read_csv(CLEANED_DATA_PATH)
    corpus["embedding"] = corpus.embedding.apply(eval)

    result = get_most_similar(corpus, target_clean, url_emb, num=5)

    result = get_most_diverse_articles(result)

    print(result.url.iloc[0])
    print(result.url.iloc[1])
    print(result.url.iloc[2])

    print(result.important_text.loc[0])
    print(result.important_text.loc[1])
    print(result.important_text.loc[2])
    # result = result.url.tolist()

    # output = divide_by_polarity_and_subjectivity(result, random=False)
    #
    # for k, v in output.items():
    #     if len(v) == 2:
    #         print(f"{k} :\n {v[0]}\n {v[1]}")
    #     elif len(v) == 1:
    #         print(f"{k} :\n {v[0]}")
    #
    #
    # print("OLD")
    #
    # print(result.url.iloc[0])
    # print(result.url.iloc[1])
    # print(result.url.iloc[2])
    # print(result.url.iloc[3])
    # print(result.url.iloc[4])


if __name__ == '__main__':
    url = "https://www.independent.co.uk/news/world/americas/us-politics/qanon-conspiracy-biden-robot-cnn-b1809531.html"

    main(url)
