from news_diversification.src.preprocessing_tfidf import ner_preprocessing, TFIDF_VECTORIZER_LOCATION, \
    TFIDF_MATRIX_LOCATION
from preprocessing import preprocess_target
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from result_ordering import divide_by_polarity_and_subjectivity
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import spacy
import pickle

pd.set_option('display.max_columns', 500)

NLP = spacy.load("en_core_web_lg")

NER_LOCATION = "../resources/extracted_named_entities.pickle"


def filter_similars(row, max_similarity, percent=0.2):
    similarity = row.similarities
    if similarity + percent >= max_similarity:
        return row
    else:
        return None


def determine_ner_value(entity):
    # source https://towardsdatascience.com/extend-named-entity-recogniser-ner-to-label-new-entities-with-spacy-339ee5979044
    label = entity[1]

    if label == "PERSON":
        return 1
    elif label == "ORG":
        return 0.7
    elif label == "GPE" or label == "LOC":
        return 0.5
    elif label == "EVENT":
        return 0.3
    elif label == "PRODUCT":
        return 0.1
    elif label == "NORP":
        return 0.1
    else:
        return 0.05


def target_ner_impact_evaluation(df, target_clean):
    df["ner_similarities"] = np.zeros(len(df))

    with open(NER_LOCATION, "rb") as f:
        d = pickle.load(f)

    target_doc = NLP(target_clean)
    target_entities = [(X.text, X.label_) for X in target_doc.ents]

    for entity in target_entities:
        if entity in d:
            print(entity)
            matched_articles = d[entity]
            for index in matched_articles:
                df["ner_similarities"][index] += determine_ner_value(entity)

    return df


def tfidf(target_clean, filter_result=False, ner=True):
    df = pd.read_csv("../resources/cleaned_data.csv")

    with open(TFIDF_VECTORIZER_LOCATION, "rb") as f:
        vectorizer = pickle.load(f)

    with open(TFIDF_MATRIX_LOCATION, "rb") as f:
        X = pickle.load(f)

    y = vectorizer.transform([target_clean])

    cosine_similarities = linear_kernel(y, X).flatten()[:-1]

    df["similarities"] = cosine_similarities

    if ner:
        df = target_ner_impact_evaluation(df, target_clean)

        df["similarities"] = 0.9 * df["similarities"] + 0.1 * df["ner_similarities"]

    df.sort_values(by='similarities', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if filter_result:
        max_similarity = df.iloc[0].similarities
        df = df.apply(lambda row: filter_similars(row, max_similarity), axis=1).dropna()

    output = divide_by_polarity_and_subjectivity(df, random=False)

    for k, v in output.items():
        if len(v) == 2:
            print(f"{k} :\n {v[0]}\n {v[1]}")
        else:
            print(f"{k} :\n {v[0]}")

    print("OLD")

    print(df.iloc[0].url, df.iloc[0].similarities)
    print(df.iloc[1].url, df.iloc[1].similarities)
    print(df.iloc[2].url, df.iloc[2].similarities)
    print(df.iloc[3].url, df.iloc[3].similarities)
    print(df.iloc[4].url, df.iloc[4].similarities)

def main(url):
    target_clean, publication_date = preprocess_target(url)
    import re
    target_clean = re.sub('\n', '', target_clean)
    print(target_clean)
    tfidf(target_clean, ner=True)


if __name__ == '__main__':
    url = "https://www.bbc.com/news/business-56559073"
    main(url)
