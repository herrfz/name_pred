import pickle
import asyncio
from aiohttp import web
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

origin = ('chinese', 'japanese')

class LengthExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        return DataFrame([len(x) for x in X])

    def fit(self, X, y=None, **fit_params):
        return self

class EndsWithConsonant(BaseEstimator, TransformerMixin):
    def transform(self, X, **transform_params):
        return DataFrame(X).applymap(lambda x: int(x[-1] not in ['a', 'i', 'u', 'e', 'o']))
    
    def fit(self, X, y=None, **fit_params):
        return self

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

@asyncio.coroutine
def handle(request):
    name = request.match_info.get('name', 'anon')
    return web.Response(body='<b>{name} is {origin}</b>'.format(name=name, origin=origin[clf.predict([name])]).encode('utf-8'))

@asyncio.coroutine
def init(loop):
    app = web.Application(loop=loop)
    app.router.add_route('GET', '/{name}', handle)

    srv = yield from loop.create_server(app.make_handler(), '127.0.0.1', 80)
    print('server started at http://127.0.0.1:80')
    return srv

loop = asyncio.get_event_loop()
loop.run_until_complete(init(loop))
try:
    loop.run_forever()
except KeyboardInterrupt:
    while loop.is_running():
        continue
    loop.close()

