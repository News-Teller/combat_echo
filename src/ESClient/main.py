import spacy
import pandas as pd

# Import project components
from helpers import get_articles
from esclient import ElasticsearchClient

NLP = spacy.load("en_core_web_sm")


def extract_ne(doc):
    """
    Uses spacy NER to extract named entities from article content.

    :param doc: txt, the raw text of the article
    :return: list, of tuples with the entities and the corresponding text
    """
    spacy_doc = NLP(doc)

    entities = list()
    for ent in spacy_doc.ents:
        if ent.label_ == 'ORG':
            entities.append((ent.text, ent.label_))

        elif ent.label_ == 'PERSON':
            entities.append((ent.text, ent.label_))

    print('\nFound {} entities.'.format(len(entities)))

    return entities


def get_scientific(df):
    """
    Select subset of the input data with article that have at least one doi in their references.

    :param df: pd.DataFrame, with the input data
    :return: pd.DataFrame, with the filtered data
    """
    # Select subset with doi in the references
    clean = list()
    for index, row in df.iterrows():
        for ref in row['references']:
            # having at least one doi reference
            if "//doi" in ref:
                clean.append(row)
                break

    filtered_df = pd.DataFrame(clean)

    # remove duplicates
    filtered_df = filtered_df.drop_duplicates(subset=['title'])
    print('Number of articles with at least one doi reference: {}'.format(len(filtered_df)))

    return filtered_df


if __name__ == '__main__':

    print("Starting...")
    # create client
    es = ElasticsearchClient('10.0.0.35')
    print("Connected")

    # set fetching corresponding time window
    from_ = '2021-02-18T00:00:00.000'
    to_ = '2021-02-18T23:59:00.000'
    timeframe = (from_, to_)

    # get articles
    print("Fetching articles...")
    articles = get_articles(es, timeframe)
    print('Total articles fetched: {}'.format(len(articles)))

    # transform fetched data to df
    data = pd.DataFrame(articles).rename(columns={0: 'datetime', 1: 'article'})
    data = data.join(data['article'].apply(pd.Series))
    cleaned_data = get_scientific(data)

    network = list()
    for index, row in cleaned_data.iterrows():
        extracted_entities = extract_ne(row['text'])
        print('\tFor article with title: `{}` found {} entities'.format(row['title'], len(extracted_entities)))

        for entity in extracted_entities:
            pair = {
                'index': index,
                'url': row['url'],
                'title': row['title'],
                'text': row['text'],
                'entity': entity,
            }
            network.append(pair)

    with open('ne_network_F20200911_T20200920.txt', 'w') as f:
        for item in network:
            f.write("%s\n" % item)
