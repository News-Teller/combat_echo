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
    for url_d in urls:
        url = url_d["expanded_url"]
        url_clean = preprocess_target(url)
        if url_clean is not None:
            url_emb = get_embedding(url_clean)
            corpus = pd.read_csv(CLEANED_DATA_PATH)

            corpus["embedding"] = corpus.embedding.apply(eval)

            result = get_most_similar(corpus, url_emb, num=10)

            print("Result:")
            print(result.cleaned_important_text)


def check_mentions(api, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline,
                               since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        urls = tweet.entities["urls"]
        process_urls(urls)

        # if tweet.in_reply_to_status_id is not None:
        #     continue
        # if any(keyword in tweet.text.lower() for keyword in keywords):
        #     logger.info(f"Answering to {tweet.user.name}")
        #
        #     if not tweet.user.following:
        #         tweet.user.follow()
        #
        #     api.update_status(
        #         status="Please reach us via DM",
        #         in_reply_to_status_id=tweet.id,
        #     )
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
