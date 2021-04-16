from preprocessing import preprocess_target
from preprocessing import get_embedding
from preprocessing import CLEANED_DATA_PATH
from similarity_calculation_spacy import get_most_similar
from preprocessing_fasttext import FasttextPreprocessor
from similarity_calculation_fasttext import SimilarityFasttext
import pandas as pd
import numpy as np
import time
from fetching import fetch
from preprocessing import preprocess_cached
from datetime import datetime

from result_ordering import divide_by_polarity_and_subjectivity

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

    # print(publication_date)

    print(target_clean)

    calculator = SimilarityFasttext(target_clean)

    result = calculator.get_similarities()

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
    url = "https://www.bbc.com/news/business-56559073"
    main(url)
