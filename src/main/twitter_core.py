import tweepy
import logging
import time
import pandas as pd
import os

from src.live_processing.media_filtering import perform_media_filtering, clean_domain
from src.utils.cuttly_connector import CuttlyConnector
from src.similarity_calculation.similarity_calculation_google_usc import SimilarityGoogleUsc
from src.similarity_calculation.similarity_calculation_tfidf import SimilarityTfidf
from src.live_processing.pca_diversification import get_most_diverse_articles
from src.utils.twitter_connector import TwitterConnector
from src.preprocessing.preprocessing import preprocess_target_bert, clean_text, remove_spaces
from src.preprocessing.preprocessing import get_embedding
from src.preprocessing.preprocessing import CLEANED_DATA_PATH
from src.similarity_calculation.similarity_calculation_spacy import get_most_similar
from src.similarity_calculation.similarity_calculation_bert import SimilarityTransformer
from src.similarity_calculation.similarity_calculation_fasttext import SimilarityFasttext
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

TRACKING_HASHTAG = "#covid"
SCIENTIFIC_GOOGLE_EMBEDDINGS = "../../resources/scientific_cleaned_google_usc.pickle"


def process_scientific_urls_google_usc(url_clean):
    calculator = SimilarityGoogleUsc(data_path=SCIENTIFIC_GOOGLE_EMBEDDINGS)

    result = calculator.calculate_similarity_for_target(url_clean, threshold=0.1)

    # TODO think about if I need to call get_most_diverse_articles, format the output properly

    # result = perform_media_filtering(result)
    if len(result) > 3:
        result = result.head(3)

    # result = get_most_diverse_articles(result)

    url = result.url.tolist()

    doi = result.doi.tolist()

    source = result.source_x.tolist()

    result = list(zip(url, source, doi))

    return result


def process_urls_google_usc(url_clean):
    calculator = SimilarityGoogleUsc()

    result = calculator.calculate_similarity_for_target(url_clean, threshold=0.1)

    result = perform_media_filtering(result)

    result = get_most_diverse_articles(result)

    url = result.url.tolist()

    bias = result.bias.tolist()

    fact = result.fact.tolist()

    result = list(zip(url, fact, bias))

    return result


def process_urls_tfidf(url, url_clean):
    calculator = SimilarityTfidf()

    result = calculator.calculate_similarities_for_target(url, url_clean)

    result = result.url.tolist()

    return result


def process_urls_bert(url_clean):
    transformer = SimilarityTransformer()

    result = transformer.calculate_similarity_for_target(url_clean)

    result = get_most_diverse_articles(result, embedding_column="bert_embedding")

    result = result.url.tolist()

    return result


def process_urls_fasttext(url_clean):
    calculator = SimilarityFasttext()

    result = calculator.get_similarities(url_clean)

    result = get_most_diverse_articles(result, embedding_column="fasttext_embedding")

    result = result.url.tolist()

    return result


def process_urls_spacy(url_clean):
    url_emb = get_embedding(url_clean)
    corpus = pd.read_csv(CLEANED_DATA_PATH)
    corpus["embedding"] = corpus.embedding.apply(eval)

    result = get_most_similar(corpus, url_clean, url_emb, num=5)

    result = get_most_diverse_articles(result)

    result = result.url.tolist()

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


def processing(text_clean, text=None, model="google_usc", scientific=False):
    if scientific:
        logger.info("Using scientific google usc")
        result = process_scientific_urls_google_usc(text_clean)
    elif model == "fasttext":
        logger.info("Using fasttext")
        result = process_urls_fasttext(text_clean)
    elif model == "bert":
        logger.info("Using bert")
        result = process_urls_bert(text_clean)
    elif model == "tfidf":
        logger.info("Using tfidf")
        result = process_urls_tfidf(text, text_clean)
    elif model == "google_usc":
        logger.info("Using google usc")
        result = process_urls_google_usc(text_clean)
    else:
        logger.info("Using spacy")
        result = process_urls_spacy(text_clean)
    return result


def process_text(text, model="google_usc", scientific=False):
    text_clean = clean_text(remove_spaces(text))
    result = processing(text_clean, model, scientific=scientific)
    result.reverse()
    return result


def process_urls(urls, model="google_usc", scientific=False):
    similar_urls = []

    for url_d in urls:
        url = url_d["expanded_url"]
        url, publication_date = preprocess_target_bert(url)  # TODO factor date
        if url is not None:
            url_clean = clean_text(remove_spaces(url))
            if url_clean is not None:
                result = processing(url_clean, url, model=model, scientific=scientific)
                result.reverse()
                similar_urls += result
    return similar_urls


def filter_out_irrelevant_urls(urls):
    if len(urls) != 0:
        filtered = list(filter(lambda url_d: clean_domain(url_d["expanded_url"]) != "twitter.com", urls))
        return filtered
    else:
        return urls


def prepare_status(tweet, similar_urls, scientific=False):
    cuttly_connector = CuttlyConnector()

    attributes = ("url", "medium bias", "medium reliability")

    if scientific:
        attributes = ("url", "source", "doi")

    status = f"""
        Hey @{tweet.user.screen_name}:    

                                {attributes[0]} | {attributes[1]} | {attributes[2]}

        1) {cuttly_connector.shorten_link(similar_urls[0][0])} | {similar_urls[0][1]} | {similar_urls[0][2]}
        2) {cuttly_connector.shorten_link(similar_urls[1][0])} | {similar_urls[1][1]} | {similar_urls[1][2]}
        3) {cuttly_connector.shorten_link(similar_urls[2][0])} | {similar_urls[2][1]} | {similar_urls[2][2]}
        """
    return status


def reply_to_user(similar_urls, api, tweet, scientific=False):
    if api.me().id != tweet.user.id:
        if not tweet.user.following:
            tweet.user.follow()
    if len(similar_urls) == 0:
        status = f"""
        Hmmmm @{tweet.user.screen_name} it seems like you have found my weakness.
        Right now, meaning {datetime.today().strftime('%d-%m-%Y-%H:%M:%S')}, I am not sure how to process this tweet
        """
    else:
        status = prepare_status(tweet, similar_urls, scientific)
    # print(status)
    api.update_status(
        status=status,
        in_reply_to_status_id=tweet.id,
    )


def check_mentions(api, since_id, model="google_usc"):
    logger.info("Retrieving mentions")

    new_since_id = since_id

    for tweet in tweepy.Cursor(api.mentions_timeline, since_id=since_id).items():

        new_since_id = max(tweet.id, new_since_id)
        original_tweet = None

        if tweet.in_reply_to_status_id is not None:  # I am mentioned in a reply
            original_tweet = api.statuses_lookup([tweet.in_reply_to_status_id])[0]
            urls = original_tweet.entities["urls"]
        else:
            urls = tweet.entities["urls"]
        urls = filter_out_irrelevant_urls(urls)

        if original_tweet is not None:
            text_to_process = original_tweet.text
        else:
            text_to_process = tweet.text

        scientific = TRACKING_HASHTAG in text_to_process

        try:

            if len(urls) == 0:
                logger.info("Processing text...")
                similar_urls = process_text(text_to_process, model=model, scientific=scientific)
            else:
                logger.info("Processing urls...")
                similar_urls = process_urls(urls, model=model, scientific=scientific)
        except Exception as e:
            logger.error(f"Exception happened while processing urls : {e}")
            similar_urls = []
        try:
            logger.info("Replying to user...")
            reply_to_user(similar_urls, api, tweet, scientific=scientific)
        except tweepy.error.TweepError as e:
            logger.info("Tweepy error")
            logger.error(f"Exact error: {e}")
            pass

    return new_since_id


def find_last_reply(api):
    my_id = api.me().id
    last_tweets = api.user_timeline(my_id)
    last_tweet = last_tweets[0]  # last tweet I posted
    reply = last_tweet.in_reply_to_status_id  # last reply I posted

    if reply is None:
        return last_tweet.id
    else:
        return reply


def main():
    curr_dir = os.getcwd()
    logger.info(f"Current directory : {curr_dir}")

    if curr_dir[-4:] != "main":
        logger.info("Changing to working directory")
        os.chdir('main')

    logger.info(f"Working at {os.getcwd()}")

    connector = TwitterConnector()
    api = connector.get_api()

    since_id = find_last_reply(api)
    # since_id = 1446819135422406655

    while True:
        since_id = check_mentions(api, since_id, model="google_usc")
        logger.info("Waiting...")
        time.sleep(10)


if __name__ == "__main__":
    main()
