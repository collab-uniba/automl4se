import json
import sys

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 autogluon/main.py github|jira")
        sys.exit(1)
    else:
        dataset = sys.argv[1]
        print("Using dataset: " + dataset)

    # load data
    ds = pd.read_csv(f"./data/{dataset}.csv")

    train, test = train_test_split(
        ds,
        train_size=0.9,
        test_size=0.1,
        random_state=1977,
        stratify=ds["Polarity"],
    )

    # train
    predictor = TabularPredictor(
        label="Polarity", problem_type="multiclass", eval_metric="f1_macro"
    ).fit(train, holdout_frac=0.2, presets="best_quality")
    predictor.leaderboard(train)

    # test
    predictions = predictor.predict(test)

    # results
    res = predictor.evaluate(test)
    cm = metrics.confusion_matrix(
        test["Polarity"], predictions, labels=["negative", "neutral", "positive"]
    )
    print(res)
    print(cm)

    with open(f"autogluon/{dataset}-res.json", encoding="utf-8", mode="w") as fp:
        json.dump(res, fp, indent=4, sort_keys=True)

    with open(f"autogluon/{dataset}-cm.txt", encoding="utf-8", mode="w") as fp:
        fp.writelines(str(cm))
