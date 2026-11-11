import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from regression_claim_audit import phrases
from regression_claim_audit.audit import audit, claims, evidence, main

EXPECTED = {'REGRESSION': ['linear regression', 'regression model', 'regression models', 'regression analysis', 'fits a regression model', 'trains a regressor', 'uses regression to predict', 'predicts a numeric value', 'predicts numeric values', 'predicts a continuous value', 'predicts continuous values', 'predicts house prices', 'predicts prices', 'predicts the price', 'predicts a price', 'estimates the value', 'estimates values', 'predicts a numeric outcome', 'predicts numeric outcomes', 'numeric prediction', 'numeric predictions', 'predicts a future value', 'predicts future values', 'predicts sales', 'predicts demand', 'predicts the score', 'predicts scores', 'predicts a rating', 'predicts ratings', 'predicts an amount', 'predicts amounts', 'predicts temperature', 'predicts energy consumption', 'regressor', 'regressors', 'regression prediction', 'regression predictions', 'predicts a target value', 'predicts target values', 'multiple regression', 'polynomial regression', 'ridge regression', 'lasso regression', 'support vector regression', 'random forest regression', 'gradient boosting regression', 'predicts a real value', 'predicts real values', 'continuous prediction', 'continuous predictions'], 'NOREGRESSION': ['no regression', 'does not do regression', 'without regression', 'no regressor', 'does not support regression', 'regression not supported', 'does not predict numeric values', 'no numeric prediction', 'cannot predict values', 'never predicts values', 'no price prediction', 'does not predict prices', 'no value estimation', 'does not estimate values', 'classification only', 'clustering only', 'no continuous prediction', 'does not predict continuous values', 'no regression model', 'does not perform regression', 'manual estimation only', 'human estimation only', 'no automated prediction', 'lookup tables only'], 'GENERIC_NAMES': ['predict_value', 'predict_numeric', 'predict_number', 'predict_continuous', 'regress', 'fit_regressor', 'predict_regression', 'regression_predict', 'estimate_value', 'estimate_price', 'predict_price', 'predict_score', 'predict_rating', 'predict_amount', 'predict_quantity', 'predict_target', 'run_regression', 'do_regression', 'apply_regression', 'predict_house_price', 'predict_sales', 'predict_demand', 'predict_temperature', 'predict_energy'], 'SKLEARN_NAMES': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'SGDRegressor', 'HuberRegressor', 'BayesianRidge', 'SVR', 'LinearSVR', 'NuSVR', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor', 'HistGradientBoostingRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'MLPRegressor', 'GaussianProcessRegressor'], 'BOOSTING_NAMES': ['XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'], 'STATSMODELS_NAMES': ['OLS', 'WLS', 'GLS', 'QuantReg']}


class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        # Fixed literal oracle, independent of implementation constants.
        tree = __import__('ast').parse(Path(phrases.__file__).read_text())
        parsed = {node.targets[0].id: __import__('ast').literal_eval(node.value)
                  for node in tree.body if isinstance(node, __import__('ast').Assign)}
        for key, value in EXPECTED.items():
            self.assertEqual(tuple(value), parsed[key])
            self.assertEqual(len(value), len(set(value)))

    def test_positive_routes(self):
        sources = [
            'from sklearn.linear_model import LinearRegression as LR; m=LR(); m.predict(X)',
            'import sklearn.linear_model as lm; m=lm.Ridge(); fitted=m.fit(X,y); fitted.predict(Z)',
            'from sklearn.linear_model import LinearRegression; LinearRegression().fit(X,y).predict(Z)',
            'from sklearn.pipeline import Pipeline; p=Pipeline(steps); p.predict(Z)',
            'from sklearn.pipeline import Pipeline; p=Pipeline(steps).fit(X,y); p.predict(Z)',
            'import xgboost as x; m=x.XGBRegressor(); m.predict(Z)',
            'from lightgbm import LGBMRegressor; LGBMRegressor().fit(X,y).predict(Z)',
            'from catboost import CatBoostRegressor as C; m=C(); f=m.fit(X,y); f.predict(Z)',
            'import statsmodels.api as sm; m=sm.OLS(y,X); result=m.fit(); result.predict(X)',
            'import statsmodels.api as sm; sm.OLS(y,X).fit().predict(X)',
            'import statsmodels.api as sm; result=sm.WLS(y,X).fit(); result.predict(X)',
            'x.predict_value(X)',
            'from service import predict_price as pp; pp(X)',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(evidence(source))
        for name in EXPECTED['SKLEARN_NAMES']:
            self.assertTrue(evidence('from sklearn import '+name+'; m='+name+'(); m.predict(X)'))
        for name in EXPECTED['BOOSTING_NAMES']:
            self.assertTrue(evidence('import xgboost as x; x.'+name+'().fit(X,y).predict(Z)'))
        for name in EXPECTED['STATSMODELS_NAMES']:
            self.assertTrue(evidence('import statsmodels.api as sm; sm.'+name+'(y,X).fit().predict(X)'))
        for name in EXPECTED['GENERIC_NAMES']:
            self.assertTrue(evidence('x.'+name+'(X)'))
            self.assertTrue(evidence('from service import '+name+' as f; f(X)'))

    def test_negative_routes(self):
        sources = [
            'import sklearn, xgboost, lightgbm, catboost, statsmodels',
            'from sklearn import LinearRegression; m=LinearRegression()',
            'from sklearn import Ridge; m=Ridge(); m.fit(X,y)',
            'import sklearn; unbound.predict(X)',
            'from sklearn import LogisticRegression; m=LogisticRegression(); m.fit(X,y); m.predict(X)',
            'from sklearn import RandomForestClassifier; m=RandomForestClassifier(); m.predict(X)',
            'from xgboost import XGBRegressor; m=XGBRegressor()',
            'import statsmodels.api as sm; m=sm.OLS(y,X); r=m.fit()',
            'import statsmodels.api as sm; r=sm.OLS(y,X).fit(); r.params',
            'import statsmodels.api as sm; sm.OLS(y,X).fit().summary()',
            'import statsmodels.api as sm; sm.Logit(y,X).fit().predict(X)',
            'from sklearn import KMeans; KMeans().fit_predict(X)',
            'from sklearn import IsolationForest; IsolationForest().fit_predict(X)',
            'from prophet import Prophet; Prophet().fit(data)',
            'from statsmodels.tsa.arima.model import ARIMA; ARIMA(data).fit().predict()',
            'from sklearn.metrics import r2_score, mean_squared_error; r2_score(y,z); mean_squared_error(y,z)',
            'from sklearn.model_selection import train_test_split; train_test_split(X)',
            'from sklearn.preprocessing import StandardScaler; StandardScaler().fit_transform(X)',
            'import pandas as pd; pd.read_csv(path)',
            'import subprocess; subprocess.run(["external-cli"])',
            'predict_price(X)', 'x.predict_value', '"x.predict_value(X)"', 'invalid !!!',
            'from service import predict_price\ndef predict_price(x): pass\npredict_price(X)',
            'def predict_value(x): pass\nx.predict_value(X)',
            'from sklearn import Ridge\nclass Ridge: pass\nm=Ridge(); m.predict(X)',
            'import xgboost as x\nclass XGBRegressor: pass\nx.XGBRegressor().fit(X,y).predict(X)',
            'import statsmodels.api as sm\ndef OLS(): pass\nsm.OLS(y,X).fit().predict(X)',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def fixture(self, doc, files):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root/'README.md').write_text(doc)
        for name, content in files.items():
            path=root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return root

    def cli(self, root, *args):
        output=io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code=main(['audit',str(root),'--json',*args])
        return code,json.loads(output.getvalue())

    def test_findings_and_thresholds(self):
        good='x.predict_value(X)'
        for doc, files, rule, severity, code in [
            ('predicts prices', {'app.py':good}, None,None,0),
            ('predicts prices', {'app.py':'pass'},'documented-regression-claim-false','HIGH',2),
            ('predicts prices', {'tests/test_model.py':good},'documented-regression-claim-false','MEDIUM',0),
            ('no regression', {'app.py':good},'documented-no-regression-claim-false','HIGH',2),
            ('Welcome', {'app.py':good},None,None,0),
            ('predicts prices', {},'documented-regression-claim-false','HIGH',2),
            ('no regression', {},None,None,0),
        ]:
            with self.subTest(doc=doc,files=files):
                root=self.fixture(doc,files)
                actual,result=self.cli(root)
                self.assertEqual(code,actual)
                self.assertIn('verdict',result)
                self.assertEqual(1 if rule else 0,len(result['findings']))
                if rule:
                    f=result['findings'][0]
                    self.assertEqual(rule,f['rule']); self.assertEqual(severity,f['severity'])
                    self.assertIn('message',f)
                    self.assertEqual(2,self.cli(root,'--fail-on','medium')[0])

    def test_document_matching(self):
        self.assertEqual(([],[]),claims('```sh\npredicts prices\n```\n~~~\nno regression\n~~~'))
        self.assertTrue(claims('`predicts prices`')[0])
        self.assertEqual(([],['does not do regression']),claims('does not do regression'))
        self.assertTrue(claims("doesn't predict prices")[1])
        self.assertTrue(claims('predicts-house-prices')[0])
        self.assertFalse(claims('predicts\n\nprices')[0])
        self.assertFalse(claims('regression testing; does regression; perform regression; logistic regression')[0])
        self.assertFalse(claims('preregressor')[0])

    def test_exclusions_and_scope(self):
        for name in ['fixtures/model.py','docs/model.py','examples/model.py']:
            root=self.fixture('predicts prices',{name:'x.predict_value(X)'})
            self.assertEqual('HIGH',audit(root)['findings'][0]['severity'])
        for name in ['tests/model.py','test_model.py','model_test.py']:
            root=self.fixture('predicts prices',{name:'x.predict_value(X)'})
            self.assertEqual('MEDIUM',audit(root)['findings'][0]['severity'])
        root=self.fixture('predicts prices',{'one.py':'from sklearn import Ridge; m=Ridge()', 'two.py':'m.predict(X)'})
        self.assertEqual('HIGH',audit(root)['findings'][0]['severity'])
        root=self.fixture('predicts prices',{'real.py':'x.predict_value(X)'})
        (root/'linked.py').symlink_to(root/'real.py')
        (root/'linked_dir').symlink_to(root, target_is_directory=True)
        self.assertEqual('HIGH',audit(root, ['real.py'])['findings'][0]['severity'])
        root=self.fixture('Welcome',{'docs/nested/guide.md':'predicts prices'})
        self.assertEqual(1,len(audit(root)['findings']))
        self.assertEqual([],audit(root,['docs/'])['findings'])

    def test_console_script_entry_point(self):
        script = __import__('shutil').which('regression-claim-audit')
        if script is None:
            self.skipTest('console script not installed in this environment')
        result = subprocess.run([script, '--version'], capture_output=True, text=True)
        self.assertEqual(0, result.returncode)
        self.assertEqual('0.1.0', result.stdout.strip())

    def test_entry_points_and_usage(self):
        result=subprocess.run([sys.executable,'-m','regression_claim_audit','--version'],capture_output=True,text=True)
        self.assertEqual(0,result.returncode); self.assertEqual('0.1.0',result.stdout.strip())
        root=self.fixture('Welcome',{})
        result=subprocess.run([sys.executable,'-m','regression_claim_audit','audit',str(root)],capture_output=True,text=True)
        self.assertEqual(0,result.returncode)
        for args in [[],['audit','--fail-on','HIGH'],['audit','--ignore','a','b','c'],['audit','/nonexistent-path-rca']]:
            with contextlib.redirect_stderr(io.StringIO()):
                try: code=main(args)
                except SystemExit as exc: code=exc.code
            self.assertEqual(1,code)


if __name__ == '__main__':
    unittest.main()
