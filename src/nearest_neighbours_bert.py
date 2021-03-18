from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch


class SimilarityTransformer:
    __CLEANED_DATA_PATH_BERT = "../resources/cleaned_data_bert.csv"
    __EMBEDDINGS_PATH = "../resources/embeddings.pt"

    def __init__(self, model_name="stsb-roberta-large"):
        self.model = self.__load_model(model_name)
        self.data = self.__load_data_with_embeddings()
        self.target_emb = None
        self.embeddings = torch.load(self.__EMBEDDINGS_PATH)

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
        return pd.read_csv(self.__CLEANED_DATA_PATH_BERT)

    def calculate_similarity_for_target(self, target, num=5):
        self.__create_embeddings_target(str(target))
        self.__calculate_similarities()
        return self.data.nlargest(num, 'similarities')
