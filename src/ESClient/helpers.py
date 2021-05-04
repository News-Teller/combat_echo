#!/usr/bin/env python

from dateutil.parser import parse
from news_diversification.src.ESClient.esclient import ElasticsearchClient


def get_articles(client: ElasticsearchClient, timeframe: tuple) -> list:
    """
    Get entities per article in a given timeframe.

    :param client: ElasticSearchClient object
    :param timeframe: Timeframe
    :return: Entities per article
    """
    generator = client.batch_query_articles(timeframe)

    articles = []

    for batch in generator:
        for document in batch:
            published_datetime = parse(document['_source']['publish_datetime'])
            article = document['_source']
            article['id'] = document['_id']
            articles.append([published_datetime, article])

    # Sort them for convinience
    # articles = sorted(articles, key=itemgetter(0), reverse=True)

    return articles
