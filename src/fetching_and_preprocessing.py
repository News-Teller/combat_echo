import spacy
import pandas as pd
import numpy as np
from ESClient.helpers import get_articles
from ESClient.esclient import ElasticsearchClient
from newsplease import NewsPlease

NLP = spacy.load("en_core_web_sm")


def convert_to_df(articles):
    data = pd.DataFrame(articles).rename(columns={0: 'datetime', 1: 'article'})
    data = data.join(data['article'].apply(pd.Series))
    data.drop("article", axis=1, inplace=True)
    return data


def fetch(from_, to_):
    print("Starting...")
    # create client
    es = ElasticsearchClient('10.0.0.35')
    print("Connected")

    timeframe = (from_, to_)

    print("Fetching articles...")
    articles = get_articles(es, timeframe)
    print('Total articles fetched: {}'.format(len(articles)))
    articles = convert_to_df(articles)
    return articles


def get_title_and_leading_paragraph_from_url(url):
    article = NewsPlease.from_url(url)
    return article.title + " " + article.description


def _get_paragraphs(row, count=1):
    full_text = row.text
    sentences = NLP(full_text)

    sentences = list(sentences.sents)  # [sent for sent in sentences.sents]  # if too slow could do with regex
    first_sentence = str(sentences[:count])
    return first_sentence.strip()


def get_title_and_leading_paragraph_from_elastic(row):
    first_sentence = _get_paragraphs(row)
    title = row.title.strip()

    merged = title + " " + first_sentence

    return merged


def preprocess_important_text(row):
    important_text = row.important_text
    important_text = important_text.lower()

    important_text = NLP(important_text)

    lemmatized = [token.lemma_ for token in important_text if not token.is_stop and not token.is_punct]

    return " ".join(lemmatized)


def preprocessing(df, news_please=False, cache = True):
    print("Parsing important text...")
    if news_please:
        df["important_text"] = df.url.apply(get_title_and_leading_paragraph_from_url)
    else:

        df["important_text"] = df.apply(get_title_and_leading_paragraph_from_elastic, axis=1)
    print("Done")

    print("Text cleaning...")

    df["cleaned_important_text"] = df.apply(preprocess_important_text, axis=1)

    print("Done")

    if cache:
        cached = pd.DataFrame(df["cleaned_important_text"])
        cached.reset_index(drop=True, inplace=True)
        np.savetxt(r'../resources/cleaned.txt', cached.values, fmt="%s")
    return df
