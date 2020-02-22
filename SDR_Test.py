
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from Options import options
from SDRClassify import SDRClassifier

# get parameters
args = options().parse()


# Data Load
try:
    df = pd.read_csv(args['data_dir'])
    if args['dataset_size'] != -1:
        df = df.sample(args['dataset_size'])
except:
    raise Exception("Data loading failed !!")


# Data Split
try:
    X = df.drop([args['target']], axis=1)
    y = df[args['target']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args['test_size'], random_state=42)
    print("\n[INFO] Dataset Size :", X.shape[0], " - Train size :",
          X_train.shape[0], " - Test size :", X_test.shape[0])
except:
    raise Exception("Data splitting failed !!")


# Parameter Optimization
if args['isOptim']:

    print("\n[INFO] - SDR Classification is training with parameters optimization ...")
    SDR = SDRClassifier()
    best_p, best_msc, best_SDR = SDR.optimizeParameters(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("\n[INFO] - Training Done ! \n P :", best_p, " MSC :", best_msc)

    pred2 = best_SDR.predict(X_test=X_test)
    acc = accuracy_score(y_test, pred2)
    f1 = f1_score(y_test, pred2, average='weighted')
    print(
        "\n[INFO] - Model Evaluation \n\nAccuracy :{0}\nF1-Score :{1}".format(acc, f1))

else:
    # Training with input parameters
    # Train Model
    print("\n[INFO] - SDR Classification is training ...")
    SDR = SDRClassifier(p=args['p'], msc=args['msc'])
    SDR.fit(X_train=X_train, y_train=y_train)
    print("\n[INFO] - Training Done !")

    # Prediction
    from time import time
    s = time()
    pred = SDR.predict(X_test=X_test)

    # Model Evaluation
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(np.array(y_test), pred.astype(float), average='weighted')
    print(
        "\n[INFO] - Model Evaluation \n\nAccuracy :{0}\nF1-Score :{1}".format(acc, f1))
