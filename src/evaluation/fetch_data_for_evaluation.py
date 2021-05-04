import pickle

import pandas as pd

from preprocessing.preprocessing import preprocess_target_bert, remove_spaces, clean_text


def fetch_data_for_evaluation(test_urls):
    urls = list(test_urls)
    df = pd.DataFrame(urls, columns=["url"])

    cleaned_texts = []
    texts = []
    dates = []

    amount = len(df)

    for i in range(amount):
        url = df["url"][i]
        print(f"Fetching article {i}/{amount} with url {url}")
        tup = None
        try:
            tup = preprocess_target_bert(url)
        except Exception as e:
            print(e)
        if tup is not None:
            target, publication_date = tup
            target_clean = remove_spaces(clean_text(target))
        else:
            target_clean, target, publication_date = "", "", None
        cleaned_texts.append(target_clean)
        texts.append(target)
        dates.append(publication_date)
        print("Done")

    print("Done fetching!")

    df["cleaned_important_text"] = cleaned_texts
    df["important_text"] = texts
    df["publish_datetime"] = dates

    df.dropna(inplace=True)

    df.to_csv("../../resources/evaluation/evaluation_cleaned_data.csv", index=False)

    return df


def process_evaluation_data(limit=10):
    df = pd.read_json("../../resources/evaluation/dataset/test.json")

    # total == 5000

    df = df.iloc[:limit]

    test_labels = set(df["label"])
    test_urls = set()
    url_mapping = {}
    for i in range(0, limit):
        s = set(df["urls"][i])
        for url in s:
            test_urls.add(url)
            small = s.copy()
            small.remove(url)
            url_mapping[url] = small

    with open("../../resources/evaluation/test_urls.pickle", 'wb') as f:
        pickle.dump(test_urls, f, pickle.HIGHEST_PROTOCOL)

    with open("../../resources/evaluation/url_mapping.pickle", 'wb') as f:
        pickle.dump(url_mapping, f, pickle.HIGHEST_PROTOCOL)

    return test_urls, url_mapping


def load_evaluation_data():
    df = pd.read_csv("../../resources/evaluation/evaluation_cleaned_data.csv")

    with open("../../resources/evaluation/test_urls.pickle", 'rb') as f:
        test_urls = pickle.load(f)

    with open("../../resources/evaluation/url_mapping.pickle", 'rb') as f:
        url_mapping = pickle.load(f)

    return df, test_urls, url_mapping


def main():
    test_urls, url_mapping = process_evaluation_data(110)

    fetch_data_for_evaluation(test_urls)


if __name__ == '__main__':
    main()
