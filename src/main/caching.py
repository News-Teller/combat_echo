from fetching import fetch
from live_processing.media_filtering import cache_media_dict
from preprocessing.preprocessing import preprocess_cached
from preprocessing.preprocessing_bert import BertPreprocessor
from preprocessing.preprocessing_fasttext import FasttextPreprocessor
import logging
import os
from preprocessing.preprocessing_google_usc import GoogleUscPreprocessor
from preprocessing.preprocessing_tfidf import TfidfPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == '__main__':

    curr_dir = os.getcwd()
    logger.info(f"Current directory : {curr_dir}")

    if curr_dir[-4:] != "main":
        logger.info("Changing to working directory")
        os.chdir('main')

    logger.info(f"Working at {os.getcwd()}")

    logger.info("Starting caching pipeline")

    from_ = '2021-05-13T00:00:00.000'
    to_ =   '2021-05-20T23:59:00.000'

    logger.info(f"Dates interval : from {from_} to {to_}")

    df = fetch(from_, to_)
    df = preprocess_cached(df)

    logger.info("TFIDF preprocessing pipeline begins")

    preprocessorTfidf = TfidfPreprocessor(df)
    preprocessorTfidf.run_pipeline()

    logger.info("TFIDF preprocessing pipeline done")

    logger.info("Google USC preprocessing pipeline begins")

    preprocessorGoogleUsc = GoogleUscPreprocessor(df)
    preprocessorGoogleUsc.calculate_embeddings_and_save()

    logger.info("Google USC preprocessing pipeline done")

    logger.info("Caching media dictionary...")

    cache_media_dict()

    logger.info("Cached media dictionary")

    logger.info("Fasttext preprocessing pipeline begins")

    preprocessorFasttext = FasttextPreprocessor(df)
    df = preprocessorFasttext.run_pipeline()

    logger.info("Fasttext preprocessing pipeline done")

    logger.info("Bert preprocessing pipeline begins")

    preprocessorBert = BertPreprocessor(df)
    df = preprocessorBert.calculate_embeddings_and_save()

    logger.info("Bert preprocessing pipeline done")

