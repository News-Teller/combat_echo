import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import spacy
import pickle


class SimilarityTfidf:
    NLP = spacy.load("en_core_web_lg")

    CLEANED_DATA_LOCATION = "../../resources/cleaned_data.csv"
    NER_LOCATION = "../../resources/extracted_named_entities.pickle"
    ENTITIES_WEIGHTS_LOCATION = "../../resources/extracted_named_entities_weights.pickle"
    TFIDF_VECTORIZER_LOCATION = "../../resources/tfidf_vectorizer.pk"
    TFIDF_MATRIX_LOCATION = "../../resources/tfidf_matrix.pk"

    def __init__(self, cleaned_data_location=None, ner_location=None, entities_weight_location=None, tfidf_vectorizer_location=None,
                 tfidf_matrix_location=None):

        if cleaned_data_location:
            self.cleaned_data_location = cleaned_data_location
        else:
            self.cleaned_data_location = self.CLEANED_DATA_LOCATION

        if ner_location:
            self.ner_location = ner_location
        else:
            self.ner_location = self.NER_LOCATION

        if entities_weight_location:
            self.entities_weight_location = entities_weight_location
        else:
            self.entities_weight_location = self.ENTITIES_WEIGHTS_LOCATION

        if tfidf_vectorizer_location:
            self.tfidf_vectorizer_location = tfidf_vectorizer_location
        else:
            self.tfidf_vectorizer_location = self.TFIDF_VECTORIZER_LOCATION

        if tfidf_matrix_location:
            self.tfidf_matrix_location = tfidf_matrix_location
        else:
            self.tfidf_matrix_location = self.TFIDF_MATRIX_LOCATION

        self.df = pd.read_csv(self.cleaned_data_location)
        self.__load_data()

    def __load_data(self):
        with open(self.tfidf_vectorizer_location, "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(self.tfidf_matrix_location, "rb") as f:
            self.X = pickle.load(f)

        with open(self.entities_weight_location, "rb") as f:
            self.ner_weights = pickle.load(f)

        with open(self.ner_location, "rb") as f:
            self.ner_dict = pickle.load(f)

    def calculate_similarities_for_target(self, target, target_clean, num=5, ner=False):
        y = self.vectorizer.transform([target_clean])
        cosine_similarities = linear_kernel(y, self.X).flatten()[:-1]
        self.df["similarities"] = cosine_similarities

        if ner:
            self.__find_entities_for_target(target)

            self.df["similarities"] = 0.9 * self.df["similarities"] + 0.1 * self.df["ner_similarities"]

        self.df.sort_values(by='similarities', ascending=False, inplace=True)

        temp = self.df[self.df["cleaned_important_text"] != target_clean]

        temp.reset_index(drop=True, inplace=True)

        return temp.head(num)

    # def tfidf(self, target, target_clean, filter_result=False, ner=True):
    #
    #     if filter_result:
    #         max_similarity = df.iloc[0].similarities
    #         df = df.apply(lambda row: self.filter_similars(row, max_similarity), axis=1).dropna()
    #
    #     return df

    def __filter_similars(self, row, max_similarity, percent=0.2):
        similarity = row.similarities
        if similarity + percent >= max_similarity:
            return row
        else:
            return None

    def __determine_ner_weight(self, entity):
        # source https://towardsdatascience.com/extend-named-entity-recogniser-ner-to-label-new-entities-with-spacy-339ee5979044
        label = entity[1]

        if label in self.ner_weights:
            return 10 * self.ner_weights[label]
        else:
            return 0

    def __find_entities_for_target(self, target_clean):
        self.df["ner_similarities"] = np.zeros(len(self.df))

        target_doc = self.NLP(target_clean)
        target_entities = [(X.text, X.label_) for X in target_doc.ents]

        for entity in target_entities:
            if entity in self.ner_dict:
                matched_articles = self.ner_dict[entity]
                for index in matched_articles:
                    self.df["ner_similarities"][index] += self.__determine_ner_weight(entity)
