from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from datetime import datetime
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilarityGoogleUsc:
    __CLEANED_DATA_PATH_GOOGLE_USC = "../../resources/cleaned_google_usc.pickle"

    def __init__(self, data_path=None):
        self.embed = hub.load("../../resources/universal-sentence-encoder_4")

        if data_path:
            self.data_path = data_path
        else:
            self.data_path = self.__CLEANED_DATA_PATH_GOOGLE_USC

        self.data = self.__load_data_with_embeddings()
        self.target_emb = None

    def __calculate_similarities(self):

        embeddings = list(self.data.embedding)

        sims = cosine_similarity(np.array(self.target_emb).reshape(1, -1), embeddings)

        pairs = []
        for i in range(sims.shape[1]):
            pairs.append({'index': i, 'score': sims[0][i]})

        self.data["similarities"] = [pair["score"].item() for pair in pairs]

    def __create_embeddings_target(self, target):
        self.target_emb = self.embed([target])

    def __load_data_with_embeddings(self):
        df = pd.read_pickle(self.data_path)
        # TODO figure out what to do with date
        # df["publish_datetime"] = df["publish_datetime"].apply(
        #     lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.000Z'))
        return df

    @staticmethod
    def __threshold_similarity(df, threshold=0.1):
        max_similarity = df.similarities.loc[0]
        min_similarity = max_similarity - threshold * max_similarity

        threshold_df = df[df["similarities"] >= min_similarity]

        if len(threshold_df) < 5:
            return df.head(5)
        else:
            return threshold_df

    def calculate_similarity_for_target(self, target, threshold=0.1):
        self.__create_embeddings_target(str(target))
        self.__calculate_similarities()

        # self.data.drop_duplicates("similarities", inplace=True)
        copy = self.data.sort_values(by='similarities', ascending=False)

        copy = copy[copy["important_text"] != target]

        copy.reset_index(drop=True, inplace=True)

        copy = self.__threshold_similarity(copy, threshold)

        return copy
