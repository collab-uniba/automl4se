import json
import sys

import tensorflow as tf
from sklearn import metrics
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
    dataset = tf.keras.utils.get_file(
        fname=f"{dataset}.csv",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=False,
    )

    # prepare data
    x = dataset.drop(columns="Polarity", axis=1, inplace=False)
    y = dataset["solution"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=1977, stratify=y
    )

    # train
    clf = ak.TextClassifier(
        overwrite=True, max_trials=1
    )  # It only tries 1 model as a quick demo.
    clf.fit(
        x_train,
        y_train,
        # Split the training data and use the last 15% as validation data.
        validation_split=0.2,
    )

    # Predict with the best model.
    predicted_y = clf.predict(x_test)
    # Evaluate the best model with testing data.
    res = clf.evaluate(x_test, y_test)
    cm = metrics.confusion_matrix(
        y_test, predicted_y, labels=["negative", "neutral", "positive"]
    )

    # get results
    print(res)
    print(cm)

    with open(f"autokeras/{dataset}-res.json", "w") as fp:
        json.dump(res, fp, indent=4, sort_keys=True)

    with open(f"autokeras/{dataset}-cm.txt", "w") as fp:
        fp.writelines(str(cm))