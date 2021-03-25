from fetching import fetch
from preprocessing import preprocess_cached
from preprocessing_bert import BertPreprocessor
from preprocessing_fasttext import FasttextPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == '__main__':
    logger.info("Starting caching pipeline")

    from_ = '2021-03-17T00:00:00.000'
    to_ = '2021-03-24T23:59:00.000'

    logger.info(f"Dates interval : from {from_} to {to_}")

    df = fetch(from_, to_)
    df = preprocess_cached(df)

    logger.info("Fasttext preprocessing pipeline begins")

    preprocessorFasttext = FasttextPreprocessor(df)
    df = preprocessorFasttext.run_pipeline()

    logger.info("Fasttext preprocessing pipeline done")

    logger.info("Bert preprocessing pipeline begins")

    preprocessorBert = BertPreprocessor(df)
    df = preprocessorBert.calculate_embeddings_and_save()

    logger.info("Bert preprocessing pipeline done")

