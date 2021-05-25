import logging
import tensorflow_hub as hub


class GoogleUscPreprocessor:
    __CLEANED_DATA_PATH_GOOGLE_USC = "../../resources/cleaned_google_usc.pickle"
    logging.basicConfig(level=logging.INFO)
    __logger = logging.getLogger()

    def __init__(self, data, data_path=None):
        self.embed = hub.load("../../resources/universal-sentence-encoder_4")
        self.data = data

        if data_path:
            self.data_path = data_path
        else:
            self.data_path = self.__CLEANED_DATA_PATH_GOOGLE_USC

    def __create_embeddings(self):
        if self.data is not None:
            embeddings = self.embed(list(self.data.cleaned_important_text)).numpy().tolist()
            self.data["embedding"] = embeddings

    def calculate_embeddings_and_save(self):
        self.__logger.info("Creating google usc embeddings...")
        self.__create_embeddings()
        self.__logger.info("Done")
        self.data.to_pickle(self.data_path)
        return self.data
