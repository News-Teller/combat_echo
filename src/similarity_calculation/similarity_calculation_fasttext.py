import fasttext
import numpy as np
import pandas as pd
from scipy.spatial import distance
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityFasttext:
    CLEANED_DATA_PATH_FASTTEXT = '../../resources/cleaned_data_fasttext.pickle'
    MODEL_PATH = "../../resources/model.bin"

    def __init__(self, cleaned_data_path=None, model_path=None):
        # self.target = target
        # self.get_target_embedding()

        if cleaned_data_path:
            self.cleaned_data_path = cleaned_data_path
        else:
            self.cleaned_data_path = self.CLEANED_DATA_PATH_FASTTEXT

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.MODEL_PATH

        self.load_model_and_data()

    def load_model_and_data(self):
        self.model = fasttext.load_model(self.model_path)
        self.data = pd.read_pickle(self.cleaned_data_path)
        # TODO figure out what to do with date
        # self.data["publish_datetime"] = self.data["publish_datetime"].apply(
        #     lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.000Z'))

    @staticmethod
    def cosine_similarity(x, y):
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        return cosine_similarity(x, y)[0][0]

    def get_mean_embedding(self, text):
        emb = np.mean([self.model.get_word_vector(word) for word in text.split()], axis=0)
        return emb

    def get_target_embedding(self, target):
        # self.target_embedding = self.model.get_word_vector(str(self.target))
        # self.target_embedding = self.get_mean_embedding(str(target))
        return self.get_mean_embedding(str(target))

    def filter_similars(self, row, max_similarity, percent=0.01):
        similarity = row.similarities
        if similarity + percent * similarity >= max_similarity:
            return row
        else:
            return None

    def get_similarities(self, target, num=5):
        similarities = list()

        target_embedding = self.get_target_embedding(target)

        for row1 in self.data.iterrows():
            similarity = self.cosine_similarity(target_embedding, row1[1].fasttext_embedding)
            similarities.append(similarity)
        self.data["similarities"] = similarities
        # self.data.drop_duplicates("similarities", inplace=True)
        copy = self.data.sort_values(by='similarities', ascending=False)

        copy = copy[copy["cleaned_important_text"] != target]

        copy.reset_index(drop=True, inplace=True)

        # max_similarity = self.data.iloc[0].similarities
        # self.data = self.data.apply(lambda row: self.filter_similars(row, max_similarity), axis=1).dropna()

        return copy.head(num)


