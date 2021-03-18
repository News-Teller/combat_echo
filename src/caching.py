from fetching import fetch
from preprocessing import preprocess_cached
from preprocessing_bert import BertPreprocessor

if __name__ == '__main__':
    from_ = '2021-03-10T00:00:00.000'
    to_ = '2021-03-17T23:59:00.000'

    df = fetch(from_, to_)
    df = preprocess_cached(df)
    preprocessor = BertPreprocessor(df)
    df = preprocessor.calculate_embeddings_and_save()
    print(df.head())
    print(df.shape)
