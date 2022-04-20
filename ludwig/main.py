import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from ludwig.api import LudwigModel
import yaml
import pprint
from ludwig.automl import auto_train

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 autogluon/main.py github|jira")
        dataset = "jira"
        # sys.exit(1)
    else:
        dataset = sys.argv[1]
        print("Using dataset: " + dataset)

    # load data
    df = pd.read_csv(f"./data/{dataset}.csv", header=0)

    # prepare data
    # x = np.array(df.Text).astype(str)
    # x = df.Text
    # mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    # df = df.replace({'Polarity': mapping})
    # y = df.Polarity
    # x_train, x_test, y_train, y_test = train_test_split(
    #    x, y, test_size=0.1, random_state=1977, stratify=y
    # )

    # training_data = pd.concat([x_train, y_train], axis=1)
    # test_data = pd.concat([x_test, y_test], axis=1)

    # train a model
    config = None
    with open(r"./ludwig/config.yaml", mode="r", encoding="UTF-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    auto_train_results = auto_train(dataset=df, target="Polarity", user_config=config, time_limit_s=1800, random_seed=1977, output_directory=f"./ludwig-{dataset}")
    print(auto_train_results.best_model)
    print(auto_train_results.experiment_analysis)
    # model = LudwigModel(config)

    # train_stats = model.train(training_data)

    # obtain predictions
    # predictions = model.predict(test_data)
    # cm = metrics.confusion_matrix(
    #     y_test, predictions, labels=["negative", "neutral", "positive"]
    # )
    # cr = classification_report(y_test, predictions)

    # # get results
    # print(cr)
    # print(cm)

    # with open(f"./autokeras/{dataset}-cm.txt", encoding="utf-8", mode="w") as fp:
    #     fp.writelines(str(cm))

    # with open(f"./autokeras/{dataset}-cr.txt", encoding="utf-8", mode="w") as fp:
    #     fp.writelines(str(cr))
