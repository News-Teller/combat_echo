import spacy
import logging
from pandas.core.common import SettingWithCopyWarning
import warnings
import pandas as pd
from src.preprocessing.preprocessing import clean_text, remove_spaces

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
NLP = spacy.load("en_core_web_lg")
CLEANED_SCIENTIFIC_DATA_PATH = "../../resources/external/cleaned_scientific_data.csv"


def get_abstract(row, count=5):
    full_text = row.abstract

    sentences = NLP(full_text)

    sentences = list(sentences.sents)

    if count < len(sentences):
        first_sentence = str(sentences[:count])
        return first_sentence.strip()
    else:
        return str(sentences)


def get_title_and_abstract(row):
    abstract = get_abstract(row, count=5)
    title = row.title.strip()

    merged = title + " " + abstract

    return merged


def preprocess_scientific_cached(df, from_date=None, cache=True):
    df.dropna(inplace=True)

    if from_date:
        df["publish_time"] = pd.to_datetime(df["publish_time"], format='%Y-%m-%d')
        df = df[df["publish_time"] > from_date]

    logger.info(f"Processing {len(df)} records...")

    logger.info("Parsing important text...")

    df["important_text"] = df.apply(get_title_and_abstract, axis=1)
    logger.info("Done")

    logger.info("Text cleaning...")

    df["cleaned_important_text"] = df.important_text.apply(clean_text)

    df["cleaned_important_text"] = df.cleaned_important_text.apply(remove_spaces)

    df["important_text"] = df.important_text.apply(remove_spaces)

    df.drop_duplicates("cleaned_important_text", inplace=True)

    logger.info("Done")

    if cache:
        df.to_csv(CLEANED_SCIENTIFIC_DATA_PATH, index=False)
    return df
