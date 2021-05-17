# from result_ordering import divide_by_polarity_and_subjectivity
from preprocessing.preprocessing import preprocess_target_bert
from news_diversification.src.main.result_ordering import divide_by_polarity_and_subjectivity
from news_diversification.src.similarity_calculation.similarity_calculation_bert import SimilarityTransformer
from similarity_calculation.pca_diversification import get_most_diverse_articles


def main(url):
    target_clean, publication_date = preprocess_target_bert(url)  # TODO factor date

    # target_clean = clean_text("Hate crimes in Asia are not unique to the United States. An Asian man was brutally attacked in London during a live stream, but he received a great deal of help from an angry bystander.")

    print(target_clean)

    transformer = SimilarityTransformer()

    result = transformer.calculate_similarity_for_target(target_clean, num=10)


    result = get_most_diverse_articles(result, embedding_column="bert_embedding")

    print(result.url.iloc[0])
    print(result.url.iloc[1])
    print(result.url.iloc[2])

    print(result.important_text.loc[0])
    print(result.important_text.loc[1])
    print(result.important_text.loc[2])


if __name__ == '__main__':
    url = "https://www.independent.co.uk/news/world/americas/us-politics/qanon-conspiracy-biden-robot-cnn-b1809531.html"

    main(url)
