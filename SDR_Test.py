# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from Options import options
from SDRClassify import SDRClassifier


args = options().parse()
print(args)

#Data Load
df = pd.read_csv(args['data_dir'])
if args['dataset_size'] != -1:
    df = df.sample(args['dataset_size'])
X = df.drop([args['target']],axis=1)
y = df[args['target']]

#Data Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=args['test_size'],random_state=42)
print("\n\t Dataset Info\n\nTrain size:",X_train.shape[0],"\nTest size:",X_test.shape[0])

#Parameter Optimization
if args['isOptim']:
    SDR = SDRClassifier()
    best_p, best_msc, best_SDR = SDR.optimizeParameters(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    pred2 =  best_SDR.predict(X_test=X_test,y_test=y_test)
    acc = accuracy_score(y_test,pred2)
    f1 = f1_score(y_test,pred2,average='weighted')
    print("\n\t Model Evaluation \n\nAccuracy :{0}\nF1-Score :{1}".format(acc,f1))
else:
    #Training with input parameters
    #Train Model
    SDR = SDRClassifier(p=args['p'], msc=args['msc'])
    SDR.fit(X_train=X_train,y_train=y_train)

    #Prediction
    pred = SDR.predict(X_test=X_test,y_test=y_test)

    #Model Evaluation
    acc = accuracy_score(y_test,pred)
    f1 = f1_score(y_test,pred,average='weighted')
    print("\n\n\t Model Evaluation \n\nAccuracy :{0}\nF1-Score :{1}".format(acc,f1))
