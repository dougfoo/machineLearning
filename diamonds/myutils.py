from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## myutils
# cleanup the cols embedded w/ arrays
def cleanBracketsToF(x):
    return float(cleanBracketsToS(x))

def cleanBracketsToS(x):
    return x.replace("['",'').replace("']",'')

def cleanCut(str):
    return str[str.index('label') + 9: str.index('labelSmall')-4]

def run_linear2(data, target, norm=False, viz=True, log=True):
    # Split the targets into training/testing sets
    y_train = target[:-20]
    y_test = target[-20:]

    # Split the data into training/testing sets
    X_train = data[:-20]
    X_test = data[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression(normalize=norm)

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # Score ?
    r2score = regr.score(X_test, y_test)
    print('Score: ', r2score)
    
    # The coefficients
    if (viz):
        coeff_df = pd.DataFrame(regr.coef_.T, X_train.columns, columns=['Coefficient'])  
        coeff_df.plot(kind='bar')
        plt.show()
    
    # The mean squared error
    if (log):
        print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(y_test, y_pred))
        print (X_test.sample(n=2,random_state=1))

    f = y_test.sample(n=7,random_state=1)
    g = pd.DataFrame(data=y_pred, columns=['actual']).sample(n=7,random_state=1)
    g['predict'] = f.values 
    g['diff'] = g['actual'] - g['predict']
    g['diff%'] = (abs(g['actual'] - g['predict'] ) / abs(g['actual'])) * 100

    if (log):
        print (g)
        print (y_test.describe())
    
    return regr

def run_linear3(X_train, y_train, X_test, y_test, norm=False, viz=True, log=True):
    # Create linear regression object
    regr = linear_model.LinearRegression(normalize=norm)

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # Score ?
    r2score = regr.score(X_test, y_test)
    print('Score: ', r2score)

    # The coefficients
    if (viz):
        coeff_df = pd.DataFrame(regr.coef_.T, X_train.columns, columns=['Coefficient'])  
        coeff_df.plot(kind='bar')
        plt.show()
    
    # The mean squared error
    if (log):
        print('coefficients:', regr.coef_)
        print('intercept:',regr.intercept_)
        print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
        print('R2 Variance score: %.2f' % r2_score(y_test, y_pred))
        print (X_test.sample(n=2,random_state=1))

    g = pd.DataFrame()
    g['actual'] = y_test.sample(n=7,random_state=1)
    g['predict'] = pd.DataFrame(data=y_pred, index=y_test.index).sample(n=7,random_state=1)
    g['diff'] = g['actual'] - g['predict']
    g['diff%'] = (abs(g['actual'] - g['predict'] ) / abs(g['actual'])) * 100

    if (log):
        print (g)
        print (y_test.describe())
    
    return regr, r2score


