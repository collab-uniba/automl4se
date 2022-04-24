import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from autogoal.kb import Sentence, Seq, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.ml.metrics import *
from autogoal.search import RichLogger

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 autogoal/main.py github|jira")
        sys.exit(1)
    else:
        dataset = sys.argv[1]
        print("Using dataset: " + dataset)

    # load data
    df = pd.read_csv(f"./data/{dataset}.csv", header=0)

    # prepare data
    x = np.array(df.Text).astype(str)
    y = np.array(df.Polarity)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=1977, stratify=y
    )

    automl = AutoML(
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical, 
        random_state=1977,
        validation_split=0.2,
        errors="ignore",
    )

    automl.fit(x_train, y_train, logger=RichLogger())
    score = automl.score(x_test, y_test)
    print(f"Score: {score:0.3f}")
    predicted_y = automl.predict(x_test)

    cm = metrics.confusion_matrix(
        y_test, predicted_y, labels=["negative", "neutral", "positive"]
    )
    cr = classification_report(y_test, predicted_y)

    # get results
    print(cr)
    print(cm)

    with open(f"./autogoal/{dataset}-cm.txt", encoding="utf-8", mode="w") as fp:
        fp.writelines(str(cm))
    with open(f"./autogoal/{dataset}-cr.txt", encoding="utf-8", mode="w") as fp:
        fp.writelines(str(cr))
