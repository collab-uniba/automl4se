import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import autokeras as ak

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 autogluon/main.py github|jira")
        sys.exit(1)
    else:
        dataset = sys.argv[1]
        print("Using dataset: " + dataset)

    # load data
    df = pd.read_csv(f"./data/{dataset}.csv", header=0)

    # prepare data
    x = np.array(df.Text).astype(str)
    # mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    # df = df.replace({'Polarity': mapping})
    y = np.array(df.Polarity)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=1977, stratify=y
    )

    # train
    clf = ak.TextClassifier(
        overwrite=True  # , max_trials=1
    )  # It only tries 1 model as a quick demo.
    clf.fit(
        x_train,
        y_train,
        validation_split=0.2,
    )
    model = clf.export_model()
    model.summary()

    # Predict with the best model.
    predicted_y = clf.predict(x_test)
    # Evaluate the best model with testing data.
    res = clf.evaluate(x_test, y_test)
    cm = metrics.confusion_matrix(
        y_test, predicted_y, labels=["negative", "neutral", "positive"]
    )
    cr = classification_report(y_test, predicted_y)

    # get results
    print(res)
    print(cr)
    print(cm)

    with open(f"./autokeras/{dataset}-res.json", encoding="utf-8", mode="w") as fp:
        json.dump(res, fp, indent=4, sort_keys=True)

    with open(f"./autokeras/{dataset}-cm.txt", encoding="utf-8", mode="w") as fp:
        fp.writelines(str(cm))

    with open(f"./autokeras/{dataset}-cr.txt", encoding="utf-8", mode="w") as fp:
        fp.writelines(str(cr))
