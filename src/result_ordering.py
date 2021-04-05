import numpy as np


def get_time_difference_helper(row, target_publication_date):
    diff = target_publication_date - row.publish_datetime
    diff = diff.total_seconds()

    diff = divmod(diff, 3600)[0]
    diff = divmod(diff, 24)[0]

    return 0.9 * row.similarities + 0.1 * np.exp(-abs(diff))


def divide_by_polarity_and_subjectivity(result, target_publication_date):
    output = {}

    result["new_sims"] = result.apply(lambda x: get_time_difference_helper(x, target_publication_date),
                                      axis=1)
    result = result.sort_values(by='new_sims', ascending=False)

    output["best"] = result.iloc[0]

    result_positive = result[(result["polarity"] == "Very positive") | (result["polarity"] == "Positive")]
    result_negative = result[(result["polarity"] == "Very negative") | (result["polarity"] == "Negative")]
    result_neutral = result[(result["polarity"] == "Neutral")]

    result_objective = result[(result["subjectivity"] == "Very objective") | (result["subjectivity"] == "Objective") | (
            result["subjectivity"] == "Neutral")]
    result_subjective = result[(result["subjectivity"] == "Very subjective") | (result["subjectivity"] == "Subjective")]

    print("Positive", len(result_positive))

    # output["positive"] = tuple(result_positive.sample(n=2).url)
    # output["negative"] = tuple(result_negative.sample(n=2).url)
    # output["neutral"] = tuple(result_neutral.sample(n=2).url)
    # output["objective"] = tuple(result_objective.sample(n=2).url)
    # output["subjective"] = tuple(result_subjective.sample(n=2).url)

    output["positive"] = tuple(result_positive.iloc[0:2].url)
    output["negative"] = tuple(result_negative.iloc[0:2].url)
    output["neutral"] = tuple(result_neutral.iloc[0:2].url)
    output["objective"] = tuple(result_objective.iloc[0:2].url)
    output["subjective"] = tuple(result_subjective.iloc[0:2].url)

    return output
