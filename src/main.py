from preprocessing import preprocess_target
from preprocessing import get_embedding
from preprocessing import CLEANED_DATA_PATH
from nearest_neighbours_spacy import get_most_similar
import pandas as pd
import time


def main(url):
    start = time.time()

    target_clean = preprocess_target(url)
    target_emb = get_embedding(target_clean)
    print(target_clean)

    corpus = pd.read_csv(CLEANED_DATA_PATH)

    corpus["embedding"] = corpus.embedding.apply(eval)

    result = get_most_similar(corpus, target_emb, num=10)

    print("Result:")
    print(result.cleaned_important_text)


    end = time.time()
    print("TIME ELAPSED:")
    print(end - start)


if __name__ == '__main__':
    # url = "https://www.bloomberg.com/news/articles/2021-03-08/deliveroo-kicks-off-london-ipo-bolstering-a-busy-u-k-market?srnd=premium-europe"
    # url = "https://edition.cnn.com/2021/03/07/uk/oprah-harry-meghan-interview-intl-hnk/index.html"
    url = "https://www.bbc.com/news/uk-56329887"
    # from_ = '2021-03-05T00:00:00.000'
    # to_ = '2021-03-09T13:00:00.000'

    main(url)
