"""Read-only inventory of current executable sources; no imports or training."""
import ast
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / 'tmp/framework_reaudit_20261018'
OUT.mkdir(parents=True, exist_ok=True)
PACKAGES = ('core', 'chapter3_bser', 'chapter4_rcag', 'chapter5_vsgc', 'legacy_adapters', 'scripts', 'tools', 'docs', 'historical_verification', '文献')
records, errors, internal_imports, suspicious = [], [], [], []
modules = {}
for package in PACKAGES:
    for path in sorted((ROOT / package).rglob('*.py')):
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        try:
            source = raw.decode('utf-8-sig')
            tree = ast.parse(source, filename=relative)
            compile(tree, relative, 'exec')
        except Exception as exc:
            errors.append(dict(path=relative, error=str(exc)))
            continue
        module = relative[:-3].replace('/', '.')
        if module.endswith('.__init__'):
            module = module[:-9]
        modules[module] = (path, tree)
        records.append(dict(path=relative, lines=len(source.splitlines()), sha256=hashlib.sha256(raw).hexdigest()))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or '']
                for name in names:
                    if name.split('.')[0] in PACKAGES or name.split('.')[0] in ('CH3', 'CH4', 'CH5'):
                        internal_imports.append(dict(path=relative, line=node.lineno, module=name))
                        if name.startswith(('legacy_adapters', 'CH3', 'CH4', 'CH5')):
                            suspicious.append(dict(path=relative, line=node.lineno, kind='legacy_import', module=name))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Attribute) and ast.unparse(node.func.value) == 'sys.path':
                    suspicious.append(dict(path=relative, line=node.lineno, kind='sys_path_mutation'))

missing = [item for item in internal_imports if item['module'] not in modules and not (ROOT / item['module'].replace('.', '/')).is_dir()]
signature_findings, checked_calls = [], 0
for module, (path, tree) in modules.items():
    bindings = {n.name:(module,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    for node in tree.body:
        if not isinstance(node,ast.ImportFrom): continue
        base = node.module or ''
        if node.level:
            package = module if path.name=='__init__.py' else module.rpartition('.')[0]
            try: base = importlib.util.resolve_name('.'*node.level+base,package)
            except (ValueError,ImportError): continue
        if base not in modules: continue
        declarations = {n.name:n for n in modules[base][1].body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
        for item in node.names:
            if item.name in declarations: bindings[item.asname or item.name] = (base,declarations[item.name])
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or any(isinstance(a,ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords): continue
        function, skip = None, 0
        if isinstance(node.func,ast.Name) and node.func.id in bindings:
            function = bindings[node.func.id][1]
            if isinstance(function,ast.ClassDef):
                # A constructor can be inherited, wrapped or metaclass-driven.
                init = next((x for x in function.body if isinstance(x,ast.FunctionDef) and x.name=='__init__'),None)
                function,skip=init,1
        elif isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id in bindings:
            decl=bindings[node.func.value.id][1]
            if isinstance(decl,ast.ClassDef):
                method=next((x for x in decl.body if isinstance(x,ast.FunctionDef) and x.name==node.func.attr),None)
                if method is not None:
                    decorators=[ast.unparse(x) for x in method.decorator_list]
                    function=method
                    skip=int('classmethod' in decorators)
        if function is None or not isinstance(function,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
        args=function.args
        # Decorated signatures can intentionally differ; examine plain methods only.
        if any(ast.unparse(d) not in ('classmethod','staticmethod') for d in function.decorator_list): continue
        positional=args.posonlyargs+args.args
        accepted={x.arg for x in args.args+args.kwonlyargs}
        unexpected=[k.arg for k in node.keywords if k.arg not in accepted] if args.kwarg is None else []
        too_many=args.vararg is None and len(node.args)>len(positional)-skip
        bound={x.arg for x in positional[skip:skip+len(node.args)]} | {k.arg for k in node.keywords}
        required_pos=positional[skip:len(positional)-len(args.defaults)]
        required_kw=[x for x,d in zip(args.kwonlyargs,args.kw_defaults) if d is None]
        missing_required=[x.arg for x in required_pos+required_kw if x.arg not in bound]
        checked_calls+=1
        if unexpected or too_many or missing_required:
            signature_findings.append(dict(path=path.relative_to(ROOT).as_posix(),line=node.lineno,
                call=ast.unparse(node.func),unexpected_keywords=unexpected,too_many_positionals=too_many,
                missing_required=missing_required,
                note='static candidate; lexical shadowing and dynamic wrappers require manual review'))
counts = collections.Counter(item['path'].split('/')[0] for item in records)
json_errors, json_count = [], 0
for path in sorted((ROOT / 'configs').rglob('*.json')):
    json_count += 1
    try:
        json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except Exception as exc:
        json_errors.append(dict(path=path.relative_to(ROOT).as_posix(), error=str(exc)))
result = dict(scope=list(PACKAGES), files=len(records), lines=sum(x['lines'] for x in records),
              per_package=dict(counts), syntax_errors=errors, internal_imports=len(internal_imports),
              missing_absolute_internal_imports=missing, forbidden_patterns=suspicious,
              config_json_files=json_count, config_json_errors=json_errors,
              statically_resolved_calls=checked_calls, signature_candidates=signature_findings,
              limitations='AST/import inventory is not a proof of all runtime behavior; tests/outputs/tmp excluded', records=records)
(OUT / 'static.json').write_bytes((json.dumps(result, indent=2, ensure_ascii=False)+'\n').encode('utf-8'))
print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))
