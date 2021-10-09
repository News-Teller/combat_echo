import pandas as pd

from src.preprocessing.preprocessing_scientific import preprocess_scientific_cached
import logging
import os
from src.preprocessing.preprocessing_google_usc import GoogleUscPreprocessor

SCIENTIFIC_DATA_PATH = "../../resources/external/metadata.csv"
SCIENTIFIC_GOOGLE_EMBEDDINGS = "../../resources/scientific_cleaned_google_usc.pickle"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == '__main__':

    curr_dir = os.getcwd()
    logger.info(f"Current directory : {curr_dir}")

    if curr_dir[-4:] != "main":
        logger.info("Changing to working directory")
        os.chdir('main')

    logger.info(f"Working at {os.getcwd()}")

    logger.info("Starting caching pipeline for scientific data")

    df = pd.read_csv(SCIENTIFIC_DATA_PATH)
    df = df[["source_x", "title", "doi", "abstract", "publish_time", "authors", "journal", "url"]]

    df = preprocess_scientific_cached(df)

    logger.info("Google USC preprocessing pipeline for scientific data begins")

    preprocessorGoogleUsc = GoogleUscPreprocessor(df, data_path=SCIENTIFIC_GOOGLE_EMBEDDINGS)
    preprocessorGoogleUsc.calculate_embeddings_and_save()

    logger.info("Google USC preprocessing pipeline done")

