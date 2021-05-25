from sentence_transformers import SentenceTransformer, util
import torch
import logging


class BertPreprocessor:
    __CLEANED_DATA_PATH_BERT = "../../resources/cleaned_data_bert.csv"
    __EMBEDDINGS_PATH = "../../resources/embeddings.pt"
    logging.basicConfig(level=logging.INFO)
    __logger = logging.getLogger()

    def __init__(self, data, model_name="stsb-roberta-large", data_path=None, embeddings_path=None):
        self.model = self.__load_model(model_name)
        self.data = data
        self.target_emb = None
        self.embeddings = None
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = self.__CLEANED_DATA_PATH_BERT
        if embeddings_path:
            self.embeddings_path = embeddings_path
        else:
            self.embeddings_path = self.__EMBEDDINGS_PATH

    @staticmethod
    def __load_model(model_name="stsb-roberta-large"):
        model = SentenceTransformer(model_name)
        return model

    def __create_embeddings(self):
        if self.data is not None:
            sentences = list(self.data.important_text)
            sentences = [str(sentence) for sentence in sentences]
            self.embeddings = self.model.encode(sentences, convert_to_tensor=True)
            torch.save(self.embeddings, self.embeddings_path)
            self.data["bert_embedding"] = list(torch.chunk(self.embeddings, self.embeddings.shape[0], dim=0))

    def calculate_embeddings_and_save(self):
        self.__logger.info("Creating bert embeddings...")
        self.__create_embeddings()
        self.__logger.info("Done")
        self.data.to_csv(self.data_path, index=False)
        return self.data
