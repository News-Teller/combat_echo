from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from datetime import datetime


class SimilarityTransformer:
    __CLEANED_DATA_PATH_BERT = "../../resources/cleaned_data_bert.csv"
    __EMBEDDINGS_PATH = "../../resources/embeddings.pt"

    def __init__(self, model_name="stsb-roberta-large", data_path=None, embeddings_path=None):
        self.model = self.__load_model(model_name)

        if data_path:
            self.data_path = data_path
        else:
            self.data_path = self.__CLEANED_DATA_PATH_BERT
        if embeddings_path:
            self.embeddings_path = embeddings_path
        else:
            self.embeddings_path = self.__EMBEDDINGS_PATH

        self.embeddings = torch.load(self.embeddings_path)
        self.data = self.__load_data_with_embeddings()
        self.target_emb = None

    @staticmethod
    def __load_model(model_name="stsb-roberta-large"):
        model = SentenceTransformer(model_name)
        return model

    def __calculate_similarities(self):
        similarity_scores = None

        if self.embeddings is not None:
            similarity_scores = util.pytorch_cos_sim(self.embeddings, self.target_emb)

        if similarity_scores is not None:
            pairs = []
            for i in range(len(similarity_scores)):
                pairs.append({'index': i, 'score': similarity_scores[i][0]})
            self.data["similarities"] = [pair["score"].item() for pair in pairs]

    def __create_embeddings_target(self, target):
        self.target_emb = self.model.encode(target, convert_to_tensor=True)

    def __load_data_with_embeddings(self):
        df = pd.read_csv(self.data_path)
        df["bert_embedding"] = list(torch.chunk(self.embeddings, self.embeddings.shape[0], dim=0))
        df["bert_embedding"] = df["bert_embedding"].apply(lambda x: x.numpy()[0])
        # TODO figure out what to do with date
        # df["publish_datetime"] = df["publish_datetime"].apply(
        #     lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.000Z'))
        return df

    def __filter_similars(self, row, max_similarity, percent=0.2):
        similarity = row.similarities
        if similarity + percent * similarity >= max_similarity:
            return row
        else:
            return None

    def calculate_similarity_for_target(self, target, num=5):
        self.__create_embeddings_target(str(target))
        self.__calculate_similarities()

        # self.data.drop_duplicates("similarities", inplace=True)
        copy = self.data.sort_values(by='similarities', ascending=False)

        copy = copy[copy["important_text"] != target]

        copy.reset_index(drop=True, inplace=True)

        return copy.head(num)
