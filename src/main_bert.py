from preprocessing import preprocess_target
from nearest_neighbours_bert import SimilarityTransformer

import pandas as pd


def main(url):

    target_clean = preprocess_target(url)

    transformer = SimilarityTransformer()

    result = transformer.calculate_similarity_for_target(target_clean)

    print(result.url.iloc[0])


if __name__ == '__main__':
    # url = "https://www.bloomberg.com/news/articles/2021-03-08/deliveroo-kicks-off-london-ipo-bolstering-a-busy-u-k-market?srnd=premium-europe"
    # url = "https://edition.cnn.com/2021/03/07/uk/oprah-harry-meghan-interview-intl-hnk/index.html"
    url = "https://www.nytimes.com/interactive/2021/03/17/upshot/partisan-segregation-maps.html"
    # from_ = '2021-03-05T00:00:00.000'
    # to_ = '2021-03-09T13:00:00.000'

    main(url)
