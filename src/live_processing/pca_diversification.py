import pandas as pd
from sklearn.decomposition import PCA
from math import sqrt
import itertools
import numpy as np


def get_triangle_area(points):
    # Heron's formula

    a = np.array(points[0][0])
    b = np.array(points[1][0])
    c = np.array(points[2][0])

    indice1 = points[0][1]
    indice2 = points[1][1]
    indice3 = points[2][1]

    ab = np.linalg.norm(a - b)
    bc = np.linalg.norm(b - c)
    ac = np.linalg.norm(a - c)

    s = (ab + bc + ac) / 2

    area = sqrt(s * (s - ab) * (s - bc) * (s - ac))

    return (area, (indice1, indice2, indice3))


def get_triangle_area_two_dim(tup):
    a = tup[0][0]
    b = tup[1][0]
    c = tup[2][0]

    indice1 = tup[0][1]
    indice2 = tup[1][1]
    indice3 = tup[2][1]

    x1, y1 = a
    x2, y2 = b
    x3, y3 = c

    l1 = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    l2 = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    l3 = sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

    p = (l1 + l2 + l3) / 2
    area = sqrt(p * (p - l1) * (p - l2) * (p - l3))

    return area, (indice1, indice2, indice3)


def get_most_diverse_articles(df, embedding_column="embedding"):
    emb = pd.DataFrame(df[embedding_column])
    copy = pd.DataFrame(df[["url", "important_text", "cleaned_important_text", "bias", "fact"]])

    df = emb[embedding_column].apply(pd.Series)

    pca = PCA(0.99)
    principalComponents = pca.fit_transform(df)

    principalDf = pd.DataFrame(principalComponents)

    l = []
    for i in range(principalDf.shape[0]):
        temp = []
        for j in range(principalDf.shape[1]):
            temp.append(principalDf.loc[i][j])
        l.append((tuple(temp), i))

    combinations = list(itertools.combinations(l, 3))

    areas = [get_triangle_area(comb) for comb in combinations]

    dict_areas = dict(areas)

    positions = dict_areas[max(dict_areas.keys())]

    result = copy.iloc[list(positions)].reset_index(drop=True)

    return result
