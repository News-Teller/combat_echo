# from nearest_neighbours_fasttext import train_model
# from nearest_neighbours_fasttext import get_nns
# import time
#
#
# def main(url, from_, to_):
#     start = time.time()
#
#     df = load_data(from_, to_)
#
#     target_clean = preprocess_target(url)
#     print(target_clean)
#     train = preprocessing(df)
#
#     train_model(train)
#
#     result = get_nns(train, target_clean)
#
#     print("Result:")
#     print(result.title)
#
#     result.drop("text", axis = 1).to_csv("../resources/result.csv", index=False)
#
#     end = time.time()
#     print("TIME ELAPSED:")
#     print(end - start)
#
#
# if __name__ == '__main__':
#     #url = "https://www.bloomberg.com/news/articles/2021-03-08/deliveroo-kicks-off-london-ipo-bolstering-a-busy-u-k-market?srnd=premium-europe"
#     #url = "https://edition.cnn.com/2021/03/07/uk/oprah-harry-meghan-interview-intl-hnk/index.html"
#     url = "https://www.bbc.com/news/uk-56329887"
#     from_ = '2021-03-05T00:00:00.000'
#     to_ = '2021-03-09T13:00:00.000'
#
#     main(url, from_, to_)
