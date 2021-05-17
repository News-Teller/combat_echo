import tweepy
import logging
import time
import pandas as pd
import os

from news_diversification.src.main.result_ordering import divide_by_polarity_and_subjectivity
from news_diversification.src.similarity_calculation.similarity_calculation_google_usc import SimilarityGoogleUsc
from news_diversification.src.similarity_calculation.similarity_calculation_tfidf import SimilarityTfidf
from similarity_calculation.pca_diversification import get_most_diverse_articles
from twitter_connector import TwitterConnector
from preprocessing.preprocessing import preprocess_target, preprocess_target_bert, clean_text, remove_spaces
from preprocessing.preprocessing import get_embedding
from preprocessing.preprocessing import CLEANED_DATA_PATH
from news_diversification.src.similarity_calculation.similarity_calculation_spacy import get_most_similar
from news_diversification.src.similarity_calculation.similarity_calculation_bert import SimilarityTransformer
from news_diversification.src.similarity_calculation.similarity_calculation_fasttext import SimilarityFasttext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def process_urls_google_usc(url_clean):
    calculator = SimilarityGoogleUsc()

    result = calculator.calculate_similarity_for_target(url_clean, threshold=0.1)

    result = get_most_diverse_articles(result)

    result = result.url.tolist()

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


def processing(text_clean, text=None, model="google_usc"):
    if model == "fasttext":
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


def process_text(text, model="google_usc"):
    print(f"Text {text}")
    text_clean = clean_text(remove_spaces(text))
    print(f"Text clean {text_clean}")
    result = processing(text_clean, model)
    result.reverse()
    return result


def process_urls(urls, filter_by=False, model="google_usc"):
    similar_urls = []

    for url_d in urls:
        url = url_d["expanded_url"]
        url, publication_date = preprocess_target_bert(url)  # TODO factor date
        if url is not None:
            url_clean = clean_text(remove_spaces(url))
            if url_clean is not None:
                result = processing(url_clean, url, model=model)
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
            print("I am mentioned in a reply")
            print("Current tweet : ")
            print(tweet.text)
            original_tweet = api.statuses_lookup([tweet.in_reply_to_status_id])[0]
            print("Original tweet : ")
            print(original_tweet.text)
            urls = original_tweet.entities["urls"]
        else:
            urls = tweet.entities["urls"]

        if len(urls) == 0:
            logger.info("Processing text...")
            if original_tweet is not None:
                similar_urls = process_text(original_tweet.text, model=model)
            else:
                similar_urls = process_text(tweet.text, model=model)
        else:
            logger.info("Processing urls...")
            similar_urls = process_urls(urls, model=model, filter_by=False)
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

    while True:
        since_id = check_mentions(api, since_id, model="google_usc")
        logger.info("Waiting...")
        time.sleep(10)


if __name__ == "__main__":
    main()
