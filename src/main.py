from fetching_and_preprocessing import fetch
from fetching_and_preprocessing import preprocessing
import pandas as pd


def is_cached():
    return True


def load_data():
    if not is_cached():
        # set fetching corresponding time window
        from_ = '2021-03-05T00:00:00.000'
        to_ = '2021-03-05T12:00:00.000'
        # to_ = '2021-03-05T23:59:00.000'

        df = fetch(from_, to_)

        df.to_csv("data.csv", index=False)
    else:
        df = pd.read_csv("../resources/data.csv")

    return df


if __name__ == '__main__':
    df = load_data()

    df = preprocessing(df)

    df.to_csv("../resources/preprocessed.csv", index=False)
