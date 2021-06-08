import pandas as pd
import logging
from ESClient.esclient import ElasticsearchClient
from ESClient.helpers import get_articles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
RAW_DATA_PATH = "../resources/raw_data.csv"


def convert_to_df(articles):
    data = pd.DataFrame(articles).rename(columns={0: 'datetime', 1: 'article'})
    data = data.join(data['article'].apply(pd.Series))
    data.drop("article", axis=1, inplace=True)
    return data


def fetch(from_, to_, cache=False):
    logger.info("Starting...")
    # create client
    es = ElasticsearchClient('10.0.0.35')
    logger.info("Connected")

    timeframe = (from_, to_)

    logger.info("Fetching articles...")
    articles = get_articles(es, timeframe)
    logger.info('Total articles fetched: {}'.format(len(articles)))
    articles = convert_to_df(articles)

    if cache:
        articles.to_csv(RAW_DATA_PATH, index=False)

    return articles
