# Elasticsearch
from elasticsearch import Elasticsearch
from dateutil.parser import parse

# Logging
import logging

log = logging.getLogger(__name__)


class ElasticsearchClient:

    def __init__(self, url: str):
        """
        A wrapper class around the elasticsearch api, specified for article fetching.

        :param url: str, the host of the elasticsearch client
        """
        self.__client = Elasticsearch([url], timeout=30)

    @staticmethod
    def __get_hits(res: dict) -> dict:
        """
        Utility method that retrieves the 'hits' field.

        :param res:   Elasticsearch result
        :return:      'hits' field
        """
        return res['hits']['hits']

    @staticmethod
    def __num_hits(res: dict) -> int:
        """
        Utility method that retrieves the length of the 'hits' field.

        :param res:   Elasticsearch result
        :return:      'hits' field length
        """
        return len(ElasticsearchClient.__get_hits(res))

    def __batch_query(self, index, body, scroll):
        # Get the first set of articles and open the context.
        res = self.__client.search(index=index, body=body, scroll=scroll)
        yield self.__get_hits(res)

        # Store the scroll ID for the follwing requests.
        scroll_id = res['_scroll_id']

        while True:
            res = self.__client.scroll(scroll_id=scroll_id, scroll=scroll)

            num_hits = self.__num_hits(res)

            # If the context is exhausted, break.
            if num_hits == 0:
                break

            log.debug(f'Queried {num_hits} documents.')

            yield self.__get_hits(res)

    def batch_query_articles(self, timeframe: tuple, size: int = 1000, scroll: str = '25m') -> list:
        """
        Generator that batch queries articles in a given timeframe.

        :param timeframe:   tuple of str, the timeframe in (from, to) format
        :param size:        int, Number of articles in a result
        :param scroll:      How long the search context should be open
        :return:            Entities per articles
        """
        body = {
            'size': size,
            'query': {
                'bool': {
                    'must': [
                        {'match': {'lang': 'en'}},   # language must be english
                        # {'match': {'references': 'doi'}}  # keep only doi
                    ],
                    'filter': [
                        {'range': {'publish_datetime': {'gte': f'{timeframe[0]}||', 'lt': f'{timeframe[1]}||'}}},
                        {'range': {'scientific_count': {'gte': 1}}}  # there should be at least one scientific reference
                    ]
                }
            }
        }

        yield from self.__batch_query('articles', body, scroll)
