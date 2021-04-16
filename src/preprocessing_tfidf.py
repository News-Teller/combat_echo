from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle

NLP = spacy.load("en_core_web_lg")

NER_LOCATION = "../resources/extracted_named_entities.pickle"
TFIDF_VECTORIZER_LOCATION = "../resources/tfidf_vectorizer.pk"
TFIDF_MATRIX_LOCATION = "../resources/tfidf_matrix.pk"


def ner_preprocessing(df):
    texts = list(df["cleaned_important_text"])
    d = {}

    for i, text in enumerate(texts):
        doc = NLP(text)
        entities = [(X.text, X.label_) for X in doc.ents]
        for entity in entities:
            if entity in d:
                d[entity].append(i)
            else:
                d[entity] = [i]

    with open(NER_LOCATION, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


def tfidf_caching(df):
    vectorizer = TfidfVectorizer()

    l = list(df["cleaned_important_text"])
    l.append("")

    X = vectorizer.fit_transform(l)

    with open(TFIDF_VECTORIZER_LOCATION, 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(TFIDF_MATRIX_LOCATION, 'wb') as f:
        pickle.dump(X, f)
