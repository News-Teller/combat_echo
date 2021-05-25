import fasttext
import numpy as np
import pandas as pd
import logging


class FasttextPreprocessor:
    TRAINING_DATA_PATH = '../../resources/train.txt'
    CLEANED_DATA_PATH_FASTTEXT = '../../resources/cleaned_data_fasttext.pickle'
    MODEL_PATH = "../../resources/model.bin"
    logging.basicConfig(level=logging.INFO)
    __logger = logging.getLogger()

    def __init__(self, df, training_path=None, cleaned_data_path=None, model_path=None):
        self.data = df
        self.model = None

        if training_path:
            self.training_path = training_path
        else:
            self.training_path = self.TRAINING_DATA_PATH

        if cleaned_data_path:
            self.cleaned_data_path = cleaned_data_path
        else:
            self.cleaned_data_path = self.CLEANED_DATA_PATH_FASTTEXT

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.MODEL_PATH

    def train_model(self, save_model=True):
        self.__logger.info("Loading cleaned data...")
        cached = pd.DataFrame(self.data["cleaned_important_text"])
        cached.reset_index(drop=True, inplace=True)
        np.savetxt(self.training_path, cached.values, fmt="%s")
        self.__logger.info("Done")
        self.__logger.info("Training model...")
        self.model = fasttext.train_unsupervised(self.training_path)
        self.__logger.info("Done")

        if save_model:
            self.model.save_model(self.model_path)

    def get_mean_embedding(self, text):
        emb = np.mean([self.model.get_word_vector(word) for word in text.split()], axis=0)
        return emb

    def calculate_embeddings(self):
        if self.model is None:
            self.__logger.error("Need to train model first")
            return
        # self.data["fasttext_embedding"] = self.data.cleaned_important_text.apply(
        #     lambda text: self.model.get_word_vector(text))
        self.data["fasttext_embedding"] = self.data.cleaned_important_text.apply(
            lambda text: self.get_mean_embedding(str(text)))

        self.data.to_pickle(self.cleaned_data_path)
        return self.data

    def run_pipeline(self):
        self.train_model()
        return self.calculate_embeddings()
