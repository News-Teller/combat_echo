from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle
import pandas as pd


class TfidfPreprocessor:
    NLP = spacy.load("en_core_web_lg")

    NER_LOCATION = "../../resources/extracted_named_entities.pickle"
    ENTITIES_WEIGHTS_LOCATION = "../../resources/extracted_named_entities_weights.pickle"
    TFIDF_VECTORIZER_LOCATION = "../../resources/tfidf_vectorizer.pk"
    TFIDF_MATRIX_LOCATION = "../../resources/tfidf_matrix.pk"

    def __init__(self, df, ner_location=None, entities_weight_location=None, tfidf_vectorizer_location=None,
                 tfidf_matrix_location=None):
        self.data = df

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

    def run_pipeline(self):
        self.__ner_preprocessing()
        self.__tfidf_caching()
        self.__calculate_ner_weights()
        return self.data

    def __ner_preprocessing(self):
        texts = list(self.data["important_text"])
        self.ner_dict = {}

        for i, text in enumerate(texts):
            doc = self.NLP(text)
            entities = [(X.text, X.label_) for X in doc.ents]
            for entity in entities:
                if entity in self.ner_dict:
                    self.ner_dict[entity].append(i)
                else:
                    self.ner_dict[entity] = [i]

        with open(self.ner_location, 'wb') as f:
            pickle.dump(self.ner_dict, f, pickle.HIGHEST_PROTOCOL)

    def __tfidf_caching(self):
        vectorizer = TfidfVectorizer()

        l = list(self.data["cleaned_important_text"])
        l.append("")

        X = vectorizer.fit_transform(l)

        with open(self.tfidf_vectorizer_location, 'wb') as f:
            pickle.dump(vectorizer, f)

        with open(self.tfidf_matrix_location, 'wb') as f:
            pickle.dump(X, f)

    def __calculate_ner_weights(self):
        new_d = {}
        for key, value in self.ner_dict.items():
            kind, count = key[1], len(value)
            if kind in new_d:
                new_d[kind] += count
            else:
                new_d[kind] = count

        df = pd.DataFrame.from_dict(new_d, orient="index")
        df.reset_index(inplace=True)
        df.columns = ["kind", "count"]
        df.sort_values(by="count", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        total = sum(df["count"])
        df["perce"] = df["count"].apply(lambda x: x / total * 100)

        saved_dict = {}
        for row in df.iterrows():
            saved_dict[row[1][0]] = round(row[1][2] / 100, 4)

        with open(self.entities_weight_location, 'wb') as f:
            pickle.dump(saved_dict, f, pickle.HIGHEST_PROTOCOL)
