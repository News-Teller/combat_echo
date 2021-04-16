from news_diversification.src.similarity_calculation_spacy import get_most_similar
from result_ordering import divide_by_polarity_and_subjectivity
from preprocessing import preprocess_target
from similarity_calculation_bert import SimilarityTransformer
from preprocessing import get_embedding

import pandas as pd
CLEANED_DATA_PATH = "../resources/cleaned_data.csv"

def main(url):

    target_clean, publication_date = preprocess_target(url) # TODO factor date

    print(target_clean)

    url_emb = get_embedding(target_clean)
    corpus = pd.read_csv(CLEANED_DATA_PATH)
    corpus["embedding"] = corpus.embedding.apply(eval)

    result = get_most_similar(corpus, url_emb, num=5)
    # result = result.url.tolist()

    output = divide_by_polarity_and_subjectivity(result, random=False)

    for k, v in output.items():
        if len(v) == 2:
            print(f"{k} :\n {v[0]}\n {v[1]}")
        elif len(v) == 1:
            print(f"{k} :\n {v[0]}")


    print("OLD")

    print(result.url.iloc[0])
    print(result.url.iloc[1])
    print(result.url.iloc[2])
    print(result.url.iloc[3])
    print(result.url.iloc[4])


if __name__ == '__main__':
    url = "https://www.bbc.com/news/business-56559073"

    main(url)
