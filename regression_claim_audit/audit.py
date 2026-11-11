"""Offline AST and documentation inspection."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys

from . import __version__
from .phrases import (REGRESSION, NOREGRESSION, GENERIC_NAMES, SKLEARN_NAMES,
                      BOOSTING_NAMES, STATSMODELS_NAMES)


def normalize(text):
    return ' '.join(text.lower().replace("n't", ' not').replace('-', ' ').split())


def matches(text, phrases):
    found = []
    for phrase in sorted(phrases, key=lambda x: (-len(x), x)):
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text):
            found.append(phrase)
            text = re.sub(pattern, ' ', text)
    return found, text


def claims(text):
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        lines.append(line)
    positive, negative = [], []
    for paragraph in re.split(r'\n\s*\n', '\n'.join(lines)):
        denied, remainder = matches(normalize(paragraph), NOREGRESSION)
        asserted, _ = matches(remainder, REGRESSION)
        positive.extend(asserted)
        negative.extend(denied)
    return positive, negative


def evidence(source):
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    imports = {}
    shadow = {node.name for node in ast.walk(tree)
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.asname or alias.name] = (node.module or '') + '.' + alias.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value

    def resolve(node):
        if isinstance(node, ast.Name):
            if node.id in shadow:
                return ''
            return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            if node.attr in shadow:
                return ''
            base = resolve(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    packages = {value.split('.')[0] for value in imports.values()}

    def kind(node, seen=None):
        seen = set() if seen is None else seen
        if isinstance(node, ast.Name):
            if node.id in seen or node.id in shadow:
                return ''
            return kind(assignments.get(node.id), seen | {node.id})
        if not isinstance(node, ast.Call):
            return ''
        name = resolve(node.func)
        tail = name.split('.')[-1]
        if tail in SKLEARN_NAMES + ('Pipeline',) and 'sklearn' in packages:
            return 'pair'
        if tail in BOOSTING_NAMES and packages.intersection({'xgboost', 'lightgbm', 'catboost'}):
            return 'pair'
        if (isinstance(node.func, ast.Attribute) and tail in STATSMODELS_NAMES
                and name.startswith('statsmodels.')):
            return 'stats'
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'fit':
            base = kind(node.func.value, seen)
            return 'fitted' if base == 'stats' else base if base == 'pair' else ''
        return ''

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = resolve(node.func)
        if name.split('.')[-1] in GENERIC_NAMES:
            if isinstance(node.func, ast.Attribute) or (isinstance(node.func, ast.Name) and node.func.id in imports):
                return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'predict':
            if kind(node.func.value) in ('pair', 'fitted'):
                return True
    return False


def audit(repo='.', ignore=()):
    root = Path(repo)
    if not root.is_dir() or root.is_symlink():
        raise OSError('repository must be a readable directory')
    positive, negative, production, tests = [], [], [], []
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(root, onerror=walk_error):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
        for filename in sorted(files):
            path = Path(directory) / filename
            relative = path.relative_to(root)
            rel = relative.as_posix()
            if path.is_symlink() or any(rel.startswith(prefix) for prefix in ignore):
                continue
            parts = relative.parts
            if rel == 'README.md' or (path.suffix == '.md' and 'docs' in parts[:-1]):
                pos, neg = claims(path.read_text(encoding='utf-8'))
                positive.extend({'path': rel, 'phrase': phrase} for phrase in pos)
                negative.extend({'path': rel, 'phrase': phrase} for phrase in neg)
            if path.suffix != '.py' or set(parts[:-1]).intersection({'fixtures', 'docs', 'examples'}):
                continue
            if evidence(path.read_text(encoding='utf-8')):
                is_test = 'tests' in parts[:-1] or filename.startswith('test_') or filename.endswith('_test.py')
                (tests if is_test else production).append(rel)
    findings = []
    if positive and not production:
        findings.append({'rule': 'documented-regression-claim-false',
                         'severity': 'MEDIUM' if tests else 'HIGH',
                         'message': 'Evidence exists only in tests.' if tests else 'Documented claim has no source evidence.',
                         'claims': positive})
    if negative and production:
        findings.append({'rule': 'documented-no-regression-claim-false', 'severity': 'HIGH',
                         'message': 'Documented denial conflicts with source evidence.', 'claims': negative})
    return {'verdict': 'FAIL' if findings else 'PASS', 'findings': findings,
            'production_evidence': production, 'test_evidence': tests}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser(prog='regression-claim-audit')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    command = sub.add_parser('audit')
    command.add_argument('repo', nargs='?', default='.')
    command.add_argument('--json', action='store_true')
    command.add_argument('--verbose', action='store_true')
    command.add_argument('--ignore', action='append', default=[])
    command.add_argument('--fail-on', choices=('high', 'medium'), default='high')
    args = parser.parse_args(argv)
    try:
        result = audit(args.repo, args.ignore)
    except (OSError, UnicodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result['verdict'])
        for finding in result['findings']:
            print('{severity} {rule}: {message}'.format(**finding))
        if args.verbose:
            print(json.dumps(result, indent=2))
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in result['findings']) else 0
