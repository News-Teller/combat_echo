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
        return 5
    elif label == "ORG":
        return 3
    elif label == "GPE" or label == "LOC":
        return 3
    elif label == "EVENT":
        return 2
    elif label == "PRODUCT":
        return 1.5
    elif label == "NORP":
        return 1.5
    else:
        return 1


def ner_preprocessing(df):
    l = list(df["cleaned_important_text"])
    d = {}

    for i, text in enumerate(l):
        doc = NLP(text)
        entities = [(X.text, X.label_) for X in doc.ents]
        for entity in entities:
            if entity in d:
                d[entity].append(i)
            else:
                d[entity] = [i]

    with open(NER_LOCATION, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


def ner_calculation(df, target_clean):
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


def tfidf(target_clean, filter_result = False, ner=True):
    df = pd.read_csv("../resources/cleaned_data.csv")

    vectorizer = TfidfVectorizer()

    l = list(df["cleaned_important_text"])
    l.append(target_clean)

    X = vectorizer.fit_transform(l)

    cosine_similarities = linear_kernel(X[-1], X).flatten()[:-1]

    df["similarities"] = cosine_similarities

    df = df[df["similarities"] != 0]

    df.reset_index(drop=True, inplace=True)

    if ner:
        # ner_preprocessing(df)
        df = ner_calculation(df, target_clean)

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


def main(url):
    target_clean, publication_date = preprocess_target(url)
    tfidf(target_clean)


if __name__ == '__main__':
    url = "https://www.nytimes.com/2021/03/31/business/economy/biden-infrastructure-plan.html"
    # url = "https://www.bloomberg.com/news/articles/2021-03-18/nokia-ceo-thinks-longer-5g-cycle-gives-him-time-to-catch-up?srnd=technology-vp"
    main(url)
