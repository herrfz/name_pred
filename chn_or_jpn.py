import pickle
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

origin = ('chinese', 'japanese')

class LengthExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        return DataFrame([len(x) for x in X])

    def fit(self, X, y=None, **fit_params):
        return self

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

while True:
    try:
        test_name = input('enter a name: ')
        print(origin[clf.predict([test_name])])
    except KeyboardInterrupt:
        print('\n')
        break

