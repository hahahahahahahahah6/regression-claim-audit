# Documentation evidence auditor

A standard-library-only audit tool that compares selected documentation phrases with static source patterns. Version 0.1.0. It runs offline, sends no telemetry, and never executes inspected code.

## Installation and use

```sh
python3 -m pip install .
regression-claim-audit --version
regression-claim-audit audit . --json
python3 -m regression_claim_audit audit . --verbose
regression-claim-audit audit . --ignore vendor/ --ignore generated/ --fail-on medium
python3 -m unittest discover
```

The optional repository argument defaults to the current directory. Each ignore option accepts one path prefix, relative to the root. The failure threshold accepts lowercase high or medium and defaults to high. Exit status is 0 when no finding reaches the threshold, 2 when one does, and 1 for usage or I/O errors. JSON contains a verdict, findings with rule, severity and message fields, and evidence paths. A finding below the threshold remains visible even when the exit status is 0.

The tool reads root README.md and Markdown files under a docs path component. Fenced blocks are excluded; inline code counts. Matching ignores case, expands contractions, replaces hyphens with spaces, and collapses whitespace within blank-line-delimited paragraphs. Denial spans are removed first. Each family produces at most one finding.

```text
documented-regression-claim-false: HIGH without evidence; MEDIUM with test evidence only
documented-no-regression-claim-false: HIGH when production evidence contradicts a denial
REGRESSION with no source files: HIGH
NOREGRESSION with no source files: PASS
```

Source files are parsed with AST. Symlinks are excluded. Files in fixtures, docs, and examples do not provide evidence. Files under tests, or named test_*.py or *_test.py, provide test evidence only. Invalid syntax provides no evidence. Library imports, model creation, fitting, metrics, and external commands alone do not establish the required operation.

```text
Supported routes:
sklearn: 20 regressor constructors or Pipeline, paired with predict
xgboost / lightgbm / catboost: XGBRegressor / LGBMRegressor / CatBoostRegressor paired with predict
statsmodels: OLS / WLS / GLS / QuantReg construction, fit, then predict
24 generic operation names, including predict_value and imported predict_price
Imported aliases, fit-result indirection, and chained fit().predict() are supported.
Local function, method, and class definitions shadow matching route names.
```

## Differentiation

This tool checks documentation against numeric-model call patterns. It differs from tools concerned with these adjacent operations, which do not establish the required source evidence:

```text
whole-image labelling
sentiment analysis
topic modelling
classification
clustering
anomaly detection
time-series forecasting
text classification checkers
```

## Honest limitations

Static pattern matching is not semantic judgement. Dead code counts. Alias chains the checker cannot follow may be missed. Matching uses exact phrases, not semantics; negation depends on the negation phrase list. Pairing must occur in the same module. The tool does not inspect runtime behavior or validate model quality.

```text
Pure-torch regression with a custom nn.Module trained with MSELoss is NOT detected.
A statsmodels fitted result used only through .params or .summary() does not count.
"no regression" also matches "no regression tests" in testing prose.
"regression", "does regression", "perform regression", "performs regression",
and "logistic regression" are deliberately NOT positive phrases.
```

The fixed phrase oracle and route checks live in tests/test_audit.py. The specification is not distributed. MIT licensed.
