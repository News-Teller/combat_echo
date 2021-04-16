import fasttext
import numpy as np
import pandas as pd
from scipy.spatial import distance
from datetime import datetime


class SimilarityFasttext:
    CLEANED_DATA_PATH_FASTTEXT = '../resources/cleaned_data_fasttext.pickle'

    def __init__(self, target):
        self.target = target
        self.load_model_and_data()
        self.get_target_embedding()

    MODEL_PATH = "../resources/model.bin"

    def load_model_and_data(self):
        self.model = fasttext.load_model(self.MODEL_PATH)
        self.data = pd.read_pickle(self.CLEANED_DATA_PATH_FASTTEXT)
        self.data["publish_datetime"] = self.data["publish_datetime"].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.000Z'))

    @staticmethod
    def cosine_similarity(x, y):
        # return distance.cosine(x, y)
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def get_mean_embedding(self, text):
        emb = np.mean([self.model.get_word_vector(word) for word in text.split()], axis=0)
        return emb

    def get_target_embedding(self):
        # self.target_embedding = self.model.get_word_vector(str(self.target))
        self.target_embedding = self.get_mean_embedding(str(self.target))

    def filter_similars(self, row, max_similarity, percent=0.01):
        similarity = row.similarities
        if similarity + percent * similarity >= max_similarity:
            return row
        else:
            return None

    def get_similarities(self):
        similarities = list()

        for row1 in self.data.iterrows():
            similarity = self.cosine_similarity(self.target_embedding, row1[1].fasttext_embedding)
            similarities.append(similarity)
        self.data["similarities"] = similarities
        self.data.drop_duplicates("similarities", inplace=True)
        self.data.sort_values(by='similarities', ascending=False, inplace=True)

        # max_similarity = self.data.iloc[0].similarities
        # self.data = self.data.apply(lambda row: self.filter_similars(row, max_similarity), axis=1).dropna()

        return self.data.head(100)
