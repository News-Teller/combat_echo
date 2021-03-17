import tweepy
import logging
import time
import pandas as pd
from twitter_connector import TwitterConnector
from preprocessing import preprocess_target
from preprocessing import get_embedding
from preprocessing import CLEANED_DATA_PATH
from nearest_neighbours_spacy import get_most_similar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def process_urls(urls):
    similar_urls = []

    for url_d in urls:
        url = url_d["expanded_url"]
        url_clean = preprocess_target(url)
        if url_clean is not None:
            url_emb = get_embedding(url_clean)
            corpus = pd.read_csv(CLEANED_DATA_PATH)

            corpus["embedding"] = corpus.embedding.apply(eval)

            result = get_most_similar(corpus, url_emb, num=5)

            result = result.url.tolist()

            similar_urls += result
    return similar_urls


def reply_to_user(similar_urls, api, tweet):
    if len(similar_urls) == 0:
        print("Hmmmm are you sure you provided a link to an article?")

    else:
        status = "Here are some articles, similar to yours: \n" + "\n".join(similar_urls)
        api.update_status(
            status=status,
            in_reply_to_status_id=tweet.id,
        )


def check_mentions(api, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline,
                               since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        urls = tweet.entities["urls"]
        logger.info("Processing urls...")
        similar_urls = process_urls(urls)
        logger.info("Replying to user...")
        reply_to_user(similar_urls, api, tweet)

    return new_since_id


def main():
    connector = TwitterConnector()
    api = connector.get_api()
    since_id = 1
    while True:
        since_id = check_mentions(api, since_id)
        logger.info("Waiting...")
        time.sleep(60)


if __name__ == "__main__":
    main()
