# Combating Social Media Echo Chambers

Social media echo chambers are a phenomenon where one is only presented with online content that refects and amplifies their existing beliefs and opinions and nothing contradicting them.

This project is an attempt to solve this problem. We have developed a twitter bot that can be tagged directly on a tweet or on a reply to another tweet that contains a link to an article and will output 3 other articles that cover the same topic from different perspectives.

If the user doesn't provide a link to an article it will take as input the text of the tweet and, if it expresses an opinion, it will output 3 articles presenting different opinions on the same topic. The behavior of this feature is naturally more volatile as there's no guarantee on what the user will provide as input.

## How it works

We consider the input to our program to be a tweet from a random user containing a link to an article and the output a reply to that tweet containing 3 links to articles covering the same topic but from different perspectives.

In order to generate the output there is a very specific pipeline that is followed. We will first explain the different components that combined create this pipeline and then we will describe in what order each one is used.

It's worth noting that since we didn't know in advance which model would perform the best, we tested a lot of them. That's why each component has its own interface that is implemented by every different model that we used.

The models we used (in alphabetical order) were:

- Bert (RoBERTa)
- Fasttext
- Google Universal Sentence Encoder
- Spacy large embeddings
- TFIDF

The different components of this project are:

- The article fetching component
- The preprocessing component
- The similarity calculation component
- The live processing component

Besides these, to fascilitate development we also created an evaluation component. Although this component is not used in production settings, it is very useful to help decide what model is best in terms of accuracy, performance etc.

### Best model

After evaluating the models in the way that's described in the [evaluation](#evaluation-component) section we got the following results:

| Model name  |  Accuracy | Time per query (sec) |
|:-:|:-:|:-:|
| Bert | 57.4%  | 0.9  |
| Fasttext  | 58.2%  | 0.17  |
| **Google Universal Sentence Encoder**  | **64.3%**  | **0.018**  |
| Spacy  | 56.5%  | 0.07  |
| TFIDF  | 39.4%  | 0.05  |

In bold we see the highest performance per category and the overall best model which was Google's Universal Sentence Encoder.

### Article fetching component

In order to propose articles to the user we need to have a database with articles. In the current version this is simply a `.csv` file that we create by fetching all articles in a given timeframe from [news-teller](http://newsteller.io/)'s ElasticSearch database.

Since we're performing heavy computations during preprocessing, ideally we'd like to keep this timeframe as short as possible.

### Preprocessing component

As the name suggests this component performs all the preprocessing necessary to make processing a user query as fast as possible.

What this means for most models is that we extract all the information we need from every article, clean it and calculate its embedding offline so that when we receive a query we only need to load it instead of recomputing it each time.

### Similarity calculation component

The similarity component takes as input a link to an article and applies the same preprocessing to it as the one that has been applied to our preprocessed corpus. Then using the the calculated embeddings, it finds and returns the most similar articles to the provided one.

### Live processing component

The live processing component receives as input a list of the most similar articles to the one provided in the original query. This component has two roles. 

First it filters out any articles coming from sources that are known to be highly unreliable (according to [https://mediabiasfactcheck.com](https://mediabiasfactcheck.com)).

Its second task is to find the most diverse articles across the ones it receives as input. Even though it might seem unintuitive, this is to guarantee that there is diversity of opinions in the final links we provide to the user. 

As a reminder, the goal of this project is to offer the user articles that cover the same topic as the one they provided but from different angles. There was a plethora of ways that we could've done this, like using an external dataset to mark the political bias of each news medium and then making sure that we output articles coming from all of the political sides. We decided against this kind of options as the granularity of that information is very coarse (medium level instead of article level) and we wanted to make sure that no bias is inserted in our processing.

That's why we took a completely different approach which is 100% unsupervised. We decided to first find the most similar articles based on their content (so that we know that they refer to the same topic as the original one) and then out of these, keep the most diverse ones (meaning that their coverage will be from a different perspective). 

But how can we guarantee this diversity on an unsupervised manner? Our intuition is that the embeddings we have calculated are basically a mapping to a high dimensional space where similar articles are close to each other. Hence, since we already know that all the articles we start with are similar, if we find the ones that are the furthest away, they will be the most diverse ones. 

To do this, we took all the combinations of size 3 of the similar articles and calculated the area of the triangle they form. The triangle with the biggest area is the one that represents the triad of articles that are the most diverse with each other. To avoid the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) we used [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) to map the embeddings to a lower dimension such that we maintain 99% of the explained variance.

So to sum up, this component receives a list of articles as the input, it takes all their combinations of size 3, it calculates the area of the triangles their embeddings form and it returns the triad corresponding to the triangle with the biggest area.

### Evaluation component

This section describes how we performed the evaluation of our model in order to determine the best one. Our evaluation focused on making sure that the model finds truly similar articles (that we can later feed into our live processing component). For our data we used [this dataset from google](https://github.com/google-research-datasets/NewSHead/releases/tag/v1.0).

The data are in JSON format and they are a mapping between a topic title and a list of articles that are represented by this title, like `title -> [article1, article2, article3, article4]`.

So in order to evaluate how well our various models choose the most similar articles, we exploited the fact that for any single article in the dataset, the most similar articles to it are the ones that belong in the same list with it. So, using the notation from above, the most similar articles to `article3` are `article1`, `article2` and `article4`. That's why we transformed the dataset to represent a mapping from an article to a list of articles that are most similar to it, for example `article3 -> [article1, article2, article4]` and then we started feeding all the articles into our models and compared their output to the expected one to calculate an accuracy. For example, if the input was `article3` and the output was `[article1, article2, article7]`, the accuracy for this input is 66.66%. The total model accuracy was calculated as the mean accuracy for every input.


## Folder structure 



## Installation

- Add `export PYTHONPATH="${PYTHONPATH}:/path/to/src"` to your .barshrc or .zshrc file

## How to:

#### Evaluate a model
- Navigate to the evaluation folder and run `python evaluate_MODEL_YOU_WANT_TO_EVALUATE.py`

#### Fetch data from the ElasticSearch and run the preprocessing
- Bla bla bla

#### Deploy locally