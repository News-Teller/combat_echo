import re
from urllib.parse import urlparse
import pandas as pd
import pickle


def clean_domain(url):
    t = urlparse(url).netloc
    has_www = bool(re.search("www.[a-z0-9_-]*.[a-z]*", url))
    if has_www:
        return '.'.join(t.split('.')[1:])
    else:
        return t


def discard_biased_media(url, media_dict, facts="low"):
    if url in media_dict:
        if media_dict[url][1] == facts:
            return None
    return url


def add_bias(url, media_dict):
    if url in media_dict:
        return media_dict[url][2]
    else:
        return "unknown"


def add_fact(url, media_dict):
    if url in media_dict:
        return media_dict[url][1]
    else:
        return "unknown"


def load_media():
    media = pd.read_csv("../../resources/external/media_bias.csv")
    return media


def create_media_dict():
    d = {}

    media = load_media()

    for row in media.iterrows():
        index = row[0]
        rest = row[1]

        url_norm = rest["source_url_normalized"]
        fact = rest["fact"]
        bias = rest["bias"]

        d[url_norm] = (index, fact, bias)

    return d


def store_media_dict(d):
    with open("../../resources/external/media_dict.pickle", 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


def load_media_dict():
    with open("../../resources/external/media_dict.pickle", "rb") as f:
        d = pickle.load(f)
    return d


def cache_media_dict():
    media_dict = create_media_dict()
    store_media_dict(media_dict)


def perform_media_filtering(df):
    media_dict = load_media_dict()
    df["cleaned_domain"] = df["url"].apply(clean_domain)
    df["cleaned_domain"] = df["cleaned_domain"].apply(lambda x: discard_biased_media(x, media_dict))
    df = df.dropna().reset_index(drop=True)

    df["bias"] = df["cleaned_domain"].apply(lambda x: add_bias(x, media_dict))
    df["fact"] = df["cleaned_domain"].apply(lambda x: add_fact(x, media_dict))

    df.drop("cleaned_domain", axis=1, inplace=True)

    return df
