import pickle
import daemon
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from bottle import route, run, template

origin = ('chinese', 'japanese')

class LengthExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        return DataFrame([len(x) for x in X])

    def fit(self, X, y=None, **fit_params):
        return self

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

@route('/:name')
def index(name='unknown'):
    return template('<b>{{name}} is {{origin}}</b>', name=name, origin=origin[clf.predict([name])])

log = open('access_log', 'a')
with daemon.DaemonContext(stderr=log):
    run(host='localhost', port=80)

