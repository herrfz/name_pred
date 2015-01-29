import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

nations = ('china', 'japan')

class LengthExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        return pd.DataFrame([len(x) for x in X])

    def fit(self, X, y=None, **fit_params):
        return self

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

while True:
    test_name = input('enter a name: ')
    print(nations[clf.predict([test_name])])
