import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn import metrics

# load data
dataset = pd.read_csv("./data/github.csv")

train, test = train_test_split(
    dataset,
    train_size=0.9,
    test_size=0.1,
    random_state=1977,
    stratify=dataset["Polarity"],
)

# train
predictor = TabularPredictor(
    label="Polarity", problem_type="multiclass", eval_metric="f1_macro"
).fit(train, holdout_frac=0.2, presets='best_quality')
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
