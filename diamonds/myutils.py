from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## myutils

def run_linear2(data, target, norm=False):
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
    print('Score: ', regr.score(X_test, y_test))
    
    # The coefficients
    coeff_df = pd.DataFrame(regr.coef_.T, X_train.columns, columns=['Coefficient'])  
    coeff_df.plot(kind='bar')
    plt.show()
    
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    print (X_test.sample(n=2,random_state=1))
    f = y_test.sample(n=7,random_state=1)
    g = pd.DataFrame(data=y_pred, columns=['actual']).sample(n=7,random_state=1)
    g['predict'] = f.values
    g['diff'] = g['actual'] - g['predict']
    g['diff%'] = (abs(g['actual'] - g['predict'] ) / abs(g['actual'])) * 100
    print (g)
    print (y_test.describe())
    


