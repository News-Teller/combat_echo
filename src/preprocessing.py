import spacy
from newsplease import NewsPlease
import logging
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
NLP = spacy.load("en_core_web_lg")
CLEANED_DATA_PATH = "../resources/cleaned_data.csv"


def filter_df(df, columns_to_keep=("text", "title")):
    return df[list(columns_to_keep)]


def get_title_and_leading_paragraph_from_url(url):
    article = NewsPlease.from_url(url)
    if article.title is None or article.description is None:
        return None
    maintext = article.maintext
    cleaned = get_paragraphs_nlp(maintext)

    return article.title + " " + article.description + " " + cleaned, article.date_publish


def get_paragraphs_efficient(full_text, length=150):
    if len(full_text) > length:
        return full_text[:length]
    else:
        return full_text


def get_paragraphs_nlp(full_text, count=5):
    sentences = NLP(full_text)

    sentences = list(sentences.sents)

    if count < len(sentences):
        first_sentence = str(sentences[:count])
        return first_sentence.strip()
    else:
        logger.info("Article too short, taking full text")
        return str(sentences)


def get_paragraphs(row, count=5, efficient=False):
    full_text = row.text

    if efficient:
        return get_paragraphs_efficient(full_text)
    else:
        return get_paragraphs_nlp(full_text)


def get_title_and_leading_paragraph_from_elastic(row):
    first_sentence = get_paragraphs(row, efficient=False)
    title = row.title.strip()

    merged = title + " " + first_sentence

    return merged


def clean_text(important_text):
    important_text = important_text.lower()

    important_text = NLP(important_text)

    lemmatized = [token.lemma_ for token in important_text if not token.is_stop and not token.is_punct]

    return " ".join(lemmatized)


def get_embedding(tokens):
    tokens = NLP(tokens)
    return tokens.vector.tolist()


def preprocess_cached(df, embedding=True, cache=True):
    df = filter_df(df, ("url", "text", "title", "publish_datetime", "polarity", "subjectivity"))
    logger.info("Parsing important text...")
    df["important_text"] = df.apply(get_title_and_leading_paragraph_from_elastic, axis=1)
    logger.info("Done")

    logger.info("Text cleaning...")

    df["cleaned_important_text"] = df.important_text.apply(clean_text)

    if embedding:
        df["embedding"] = df.cleaned_important_text.apply(get_embedding)
        df = filter_df(df, ("url", "cleaned_important_text", "embedding", "publish_datetime", "polarity", "subjectivity"))
    else:
        df = filter_df(df, ("url", "cleaned_important_text", "publish_datetime", "polarity", "subjectivity"))

    df.drop_duplicates("cleaned_important_text", inplace=True)

    logger.info("Done")

    if cache:
        df.to_csv(CLEANED_DATA_PATH, index=False)
    return df


def preprocess_target(url):
    target, publication_date = get_title_and_leading_paragraph_from_url(url)
    if target is None:
        return None
    return clean_text(target), publication_date
