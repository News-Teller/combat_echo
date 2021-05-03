import tweepy
import logging
import time
import pandas as pd

from news_diversification.src.result_ordering import divide_by_polarity_and_subjectivity
from twitter_connector import TwitterConnector
from news_diversification.src.preprocessing.preprocessing import preprocess_target
from news_diversification.src.preprocessing.preprocessing import get_embedding
from news_diversification.src.preprocessing.preprocessing import CLEANED_DATA_PATH
from news_diversification.src.similarity_calculation.similarity_calculation_spacy import get_most_similar
from news_diversification.src.similarity_calculation.similarity_calculation_bert import SimilarityTransformer
from news_diversification.src.similarity_calculation.similarity_calculation_fasttext import SimilarityFasttext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def process_urls_bert(url_clean):
    transformer = SimilarityTransformer()

    result = transformer.calculate_similarity_for_target(url_clean)

    # result = result.url.tolist()

    return result


def process_urls_fasttext(url_clean):
    calculator = SimilarityFasttext()

    result = calculator.get_similarities(url_clean)

    # result = result.url.tolist()

    return result


def process_urls_spacy(url_clean):
    url_emb = get_embedding(url_clean)
    corpus = pd.read_csv(CLEANED_DATA_PATH)
    corpus["embedding"] = corpus.embedding.apply(eval)

    result = get_most_similar(corpus, url_clean, url_emb, num=5)
    # result = result.url.tolist()

    return result


def format_output(output):
    res = ""
    for k, v in output.items():
        if len(v) == 2:
            res += f"{k} :\n {v[0]}\n {v[1]}" + "\n"
        elif len(v) == 1:
            res += f"{k} :\n {v[0]}" + "\n"
        else:
            res += "No articles" + "\n"
    return res

def process_urls(urls, filter_by = True, model="fasttext"):
    similar_urls = []

    for url_d in urls:
        url = url_d["expanded_url"]
        url_clean, publication_date = preprocess_target(url)  # TODO factor date
        if url_clean is not None:
            if model == "fasttext":
                logger.info("Using fasttext")
                result = process_urls_fasttext(url_clean)
            elif model == "bert":
                logger.info("Using bert")
                result = process_urls_bert(url_clean)
            else:
                logger.info("Using spacy")
                result = process_urls_spacy(url_clean)
            if filter_by:
                output = divide_by_polarity_and_subjectivity(result, publication_date, random=False)
                output = format_output(output)
                # print("output:", output)
                similar_urls.append(output)
                print("similar:\n", similar_urls)
            else:
                result.reverse()
                similar_urls += result
    return similar_urls


def reply_to_user(similar_urls, api, tweet):
    if api.me().id != tweet.user.id:
        if not tweet.user.following:
            tweet.user.follow()
    if len(similar_urls) == 0:
        status = f"Hmmmm @{tweet.user.screen_name} are you sure you provided a link to an article?"

    else:
        status = f"Hey @{tweet.user.screen_name} here are some articles, similar to yours: \n" + "\n".join(similar_urls)
        print(status)
    # api.update_status(
    #     status=status,
    #     in_reply_to_status_id=tweet.id,
    # )


def check_mentions(api, since_id, model="fasttext"):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline,
                               since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)

        if tweet.in_reply_to_status_id is not None:
            continue  # TODO figure out what this does

        urls = tweet.entities["urls"]
        logger.info("Processing urls...")
        similar_urls = process_urls(urls, model=model)
        logger.info("Replying to user...")
        try:
            reply_to_user(similar_urls, api, tweet)
        except tweepy.error.TweepError:
            logger.info("Posted duplicate response...")
            pass

    return new_since_id


def find_last_reply(api):
    my_id = api.me().id
    last_tweets = api.user_timeline(my_id)
    return last_tweets[0].in_reply_to_status_id


def main():
    connector = TwitterConnector()
    api = connector.get_api()

    since_id = find_last_reply(api)
    since_id = 1

    while True:
        since_id = check_mentions(api, since_id, model="bert")
        logger.info("Waiting...")
        time.sleep(10)


if __name__ == "__main__":
    main()
