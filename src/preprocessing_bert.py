from sentence_transformers import SentenceTransformer, util
import torch


class BertPreprocessor:
    __CLEANED_DATA_PATH_BERT = "../resources/cleaned_data_bert.csv"
    __EMBEDDINGS_PATH = "../resources/embeddings.pt"

    def __init__(self, data, model_name="stsb-roberta-large"):
        self.model = self.__load_model(model_name)
        self.data = data
        self.target_emb = None
        self.embeddings = None

    @staticmethod
    def __load_model(model_name="stsb-roberta-large"):
        model = SentenceTransformer(model_name)
        return model

    def __create_embeddings(self):
        if self.data is not None:
            sentences = list(self.data.cleaned_important_text)
            sentences = [str(sentence) for sentence in sentences]
            self.embeddings = self.model.encode(sentences, convert_to_tensor=True)
            torch.save(self.embeddings, self.__EMBEDDINGS_PATH)
            self.data["bert_embedding"] = list(torch.chunk(self.embeddings, self.embeddings.shape[0], dim=0))

    def calculate_embeddings_and_save(self):
        print("Creating bert embeddings...")
        self.__create_embeddings()
        print("Done")
        self.data.to_csv(self.__CLEANED_DATA_PATH_BERT, index=False)
        return self.data
