from preprocessing import preprocess_target
from similarity_calculation_fasttext import SimilarityFasttext
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from result_ordering import divide_by_polarity_and_subjectivity
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import spacy

pd.set_option('display.max_columns', 500)

NLP = spacy.load("en_core_web_lg")


def filter_similars(row, max_similarity, percent=0.2):
    similarity = row.similarities
    if similarity + percent >= max_similarity:
        return row
    else:
        return None


def determine_ner_value(entity):
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



def ner(df, target_clean):
    l = list(df["cleaned_important_text"])
    # l.append(target_clean)
    # l = l[:100]
    d = {}

    df["ner_similarities"] = np.zeros(len(df))

    for i, text in enumerate(l):
        doc = NLP(text)
        entities = [(X.text, X.label_) for X in doc.ents]
        for entity in entities:
            if entity in d:
                d[entity].append(i)
            else:
                d[entity] = [i]

    target_doc = NLP(target_clean)
    target_entities = [(X.text, X.label_) for X in target_doc.ents]

    for entity in target_entities:
        if entity in d:
            print(entity)
            matched_articles = d[entity]
            for index in matched_articles:
                df["ner_similarities"][index] += determine_ner_value(entity)

    return df


def foo(target_clean):
    df = pd.read_csv("../resources/cleaned_data.csv")

    vectorizer = TfidfVectorizer()

    l = list(df["cleaned_important_text"])
    l.append(target_clean)

    X = vectorizer.fit_transform(l)

    cosine_similarities = linear_kernel(X[-1], X).flatten()[:-1]

    df["similarities"] = cosine_similarities

    df = df[df["similarities"] != 0]

    df.reset_index(drop=True, inplace=True)

    df = ner(df, target_clean)

    df["similarities"] = 0.9 * df["similarities"] + 0.1 * df["ner_similarities"]

    df.sort_values(by='similarities', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    max_similarity = df.iloc[0].similarities

    print(df.similarities.iloc[:10])

    print("before", df.shape)
    # df = df.apply(lambda row: filter_similars(row, max_similarity), axis=1).dropna()
    print("after", df.shape)

    output = divide_by_polarity_and_subjectivity(df, random=False)

    for k, v in output.items():
        if len(v) == 2:
            print(f"{k} :\n {v[0]}\n {v[1]}")
        else:
            print(f"{k} :\n {v[0]}")


def main(url):
    # from_ = '2021-03-23T00:00:00.000'
    # to_ = '2021-03-23T23:59:00.000'
    #
    # df = fetch(from_, to_)
    # df = preprocess_cached(df)
    #
    # print(df.head())

    target_clean, publication_date = preprocess_target(url)
    foo(target_clean)
    #
    # # print(publication_date)
    #
    # print(target_clean)
    #
    # calculator = SimilarityFasttext(target_clean)
    #
    # result = calculator.get_similarities()
    #
    # output = divide_by_polarity_and_subjectivity(result, publication_date, random=True)
    #
    # for k, v in output.items():
    #     if len(v) == 2:
    #         print(f"{k} :\n {v[0]}\n {v[1]}")
    #     else:
    #         print(f"{k} :\n {v[0]}")
    #
    # print("OLD")
    #
    # print(result.iloc[0].url, result.iloc[0].similarities)
    # print(result.iloc[1].url, result.iloc[1].similarities)
    # print(result.iloc[2].url, result.iloc[2].similarities)
    # print(result.iloc[3].url, result.iloc[3].similarities)
    # print(result.iloc[4].url, result.iloc[4].similarities)


if __name__ == '__main__':
    url = "https://www.nytimes.com/2021/03/31/business/economy/biden-infrastructure-plan.html"
    main(url)
