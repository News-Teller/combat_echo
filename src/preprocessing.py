import numpy as np
import spacy
import pandas as pd
from newsplease import NewsPlease

NLP = spacy.load("en_core_web_sm")
CLEANED_DATA_PATH = "../resources/cleaned_data.csv"


def filter_df(df, columns_to_keep=("text", "title")):
    return df[list(columns_to_keep)]


def get_title_and_leading_paragraph_from_url(url):
    article = NewsPlease.from_url(url)
    return article.title + " " + article.description  # TODO make sure no Nones


def get_paragraphs_efficient(full_text, length=150):
    if len(full_text) > length:
        return full_text[:length]
    else:
        return full_text


def get_paragraphs(row, count=1, efficient=False):
    full_text = row.text

    if efficient:
        return get_paragraphs_efficient(full_text)

    sentences = NLP(full_text)

    sentences = list(sentences.sents)
    first_sentence = str(sentences[:count])

    return first_sentence.strip()


def get_title_and_leading_paragraph_from_elastic(row):
    first_sentence = get_paragraphs(row, efficient=False)
    title = row.title.strip()

    merged = title + " " + first_sentence

    return merged


def clean_text(important_text):
    important_text = important_text.lower()

    important_text = NLP(important_text)

    lemmatized = [token.lemma_ for token in important_text if not token.is_stop and not token.is_punct]

    return NLP(" ".join(lemmatized))


def get_embedding(tokens):
    return tokens.vector


def preprocess_cached(df, cache=True):
    df = filter_df(df, ("text", "title"))
    print("Parsing important text...")
    df["important_text"] = df.apply(get_title_and_leading_paragraph_from_elastic, axis=1)
    print("Done")

    print("Text cleaning...")

    df["cleaned_important_text"] = df.important_text.apply(clean_text)

    df["embedding"] = df.cleaned_important_text.apply(get_embedding)

    df = filter_df(df, ("cleaned_important_text", "embedding"))

    df.drop_duplicates(subset = ["cleaned_important_text"], inplace=True)

    print("Done")

    if cache:
        df.to_csv(CLEANED_DATA_PATH, index=False)
        # cached = pd.DataFrame(df["cleaned_important_text"])
        # cached.reset_index(drop=True, inplace=True)
        # np.savetxt(r'../resources/cleaned.txt', cached.values, fmt="%s")
    return df


def preprocess_target(url):
    target = get_title_and_leading_paragraph_from_url(url)
    return clean_text(target)
