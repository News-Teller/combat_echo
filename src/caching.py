from fetching import fetch
from preprocessing import preprocess_cached

if __name__ == '__main__':
    from_ = '2021-03-16T00:00:00.000'
    to_ = '2021-03-16T23:59:00.000'

    df = fetch(from_, to_)
    df = preprocess_cached(df)
    print(df.head())
    print(df.shape)
