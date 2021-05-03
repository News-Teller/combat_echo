from result_ordering import divide_by_polarity_and_subjectivity
from news_diversification.src.preprocessing.preprocessing import clean_text, preprocess_target_bert
import tensorflow_hub as hub
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity




def main(url):
    target_clean, publication_date = preprocess_target_bert(url)  # TODO factor date

    print(target_clean)

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    df = pd.read_csv("../resources/cleaned_data.csv")
    df["publish_datetime"] = df["publish_datetime"].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.000Z'))

    sentences = list(df.important_text)
    sentences = [str(sentence) for sentence in sentences]

    embeddings = embed(sentences)

    target_clean = clean_text("Hate crimes in Asia are not unique to the United States. An Asian man was brutally attacked in London during a live stream, but he received a great deal of help from an angry bystander.")

    target_embeddings = embed([target_clean])

    print("shape of saved embeddings", embeddings.shape)
    print("target shape", target_embeddings.shape)

    sims = cosine_similarity(target_embeddings, embeddings)

    print("sims shape", sims.shape)

    pairs = []
    for i in range(sims.shape[1]):
        pairs.append({'index': i, 'score': sims[0][i]})
    print(pairs)
    df["similarities"] = [pair["score"].item() for pair in pairs]

    df.sort_values(by='similarities', ascending=False, inplace=True)

    result = df.head(100)

    # print(df.head(10))

    output = divide_by_polarity_and_subjectivity(result, publication_date, random=False)

    for k, v in output.items():
        if len(v) == 2:
            print(f"{k} :\n {v[0]}\n {v[1]}")
        else:
            print(f"{k} :\n {v[0]}")

    print("OLD")

    print(result.url.iloc[0])
    print(result.url.iloc[1])
    print(result.url.iloc[2])
    print(result.url.iloc[3])
    print(result.url.iloc[4])


if __name__ == '__main__':
    url = "https://www.theguardian.com/uk-news/2021/apr/18/the-queen-alone-how-prince-philip-death-will-change-the-future-of-the-monarchy"

    main(url)
