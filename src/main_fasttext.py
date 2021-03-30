from preprocessing import preprocess_target
from preprocessing import get_embedding
from preprocessing import CLEANED_DATA_PATH
from similarity_calculation_spacy import get_most_similar
from preprocessing_fasttext import FasttextPreprocessor
from similarity_calculation_fasttext import SimilarityFasttext
import pandas as pd
import time
from fetching import fetch
from preprocessing import preprocess_cached


def main(url):
    # from_ = '2021-03-23T00:00:00.000'
    # to_ = '2021-03-23T23:59:00.000'
    #
    # df = fetch(from_, to_)
    # df = preprocess_cached(df)
    #
    # print(df.head())

    target_clean = preprocess_target(url)

    print(target_clean)

    calculator = SimilarityFasttext(target_clean)

    result = calculator.get_similarities()

    print(result.iloc[0].url, result.iloc[0].similarities)
    print(result.iloc[1].url, result.iloc[1].similarities)
    print(result.iloc[2].url, result.iloc[2].similarities)
    print(result.iloc[3].url, result.iloc[3].similarities)
    print(result.iloc[4].url, result.iloc[4].similarities)


if __name__ == '__main__':
    # url = "https://www.bloomberg.com/news/articles/2021-03-08/deliveroo-kicks-off-london-ipo-bolstering-a-busy-u-k-market?srnd=premium-europe"
    # url = "https://edition.cnn.com/2021/03/07/uk/oprah-harry-meghan-interview-intl-hnk/index.html"
    url = "https://www.nytimes.com/interactive/2017/11/06/opinion/how-to-reduce-shootings.html"
    # from_ = '2021-03-05T00:00:00.000'
    # to_ = '2021-03-09T13:00:00.000'

    main(url)
