import tweepy
import logging
import time
import pandas as pd
from twitter_connector import TwitterConnector
from preprocessing import preprocess_target
from preprocessing import get_embedding
from preprocessing import CLEANED_DATA_PATH
from similarity_calculation_spacy import get_most_similar
from similarity_calculation_bert import SimilarityTransformer
from similarity_calculation_fasttext import SimilarityFasttext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def process_urls_bert(url_clean):
    transformer = SimilarityTransformer()

    result = transformer.calculate_similarity_for_target(url_clean)

    result = result.url.tolist()

    return result


def process_urls_fasttext(url_clean):
    calculator = SimilarityFasttext(url_clean)

    result = calculator.get_similarities()

    result = result.url.tolist()

    return result


def process_urls_spacy(url_clean):
    url_emb = get_embedding(url_clean)
    corpus = pd.read_csv(CLEANED_DATA_PATH)
    corpus["embedding"] = corpus.embedding.apply(eval)

    result = get_most_similar(corpus, url_emb, num=5)
    result = result.url.tolist()

    return result


def process_urls(urls, model="fasttext"):
    similar_urls = []

    for url_d in urls:
        url = url_d["expanded_url"]
        url_clean = preprocess_target(url)
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
    api.update_status(
        status=status,
        in_reply_to_status_id=tweet.id,
    )


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

    while True:
        since_id = check_mentions(api, since_id, model="fasttext")
        logger.info("Waiting...")
        time.sleep(10)


if __name__ == "__main__":
    main()
