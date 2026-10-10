"""Run local engineering checks and package a clean source-only release.

Never generates model training scores. Must run from the repository checkout.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def public_files():
    roots = ('src', 'configs', 'scripts', 'web', 'examples', 'tests', 'docs', 'results', '.github', 'tools')
    allowed = {'.py', '.sh', '.ps1', '.yaml', '.yml', '.html', '.md', '.json', '.toml', '.txt'}
    files = [ROOT / name for name in ('README.md', '.gitignore', 'pyproject.toml', 'requirements.txt',
                                      'requirements-train-cu118.txt', 'requirements-demo.txt', 'requirements-demo-lock.txt')]
    for name in roots:
        for path in (ROOT / name).rglob('*'):
            if path.is_file() and path.suffix in allowed:
                relative = path.relative_to(ROOT)
                if any(part in {'.idea', '__pycache__', 'demo_databases'} for part in relative.parts):
                    continue
                if relative.as_posix() == 'docs/RESUME_AND_REVIEW.md':
                    continue
                files.append(path)
    return sorted(set(path for path in files if path.is_file()))


def main():
    os.chdir(ROOT)
    if (ROOT / 'results' / 'local_validation.json').exists():
        raise FileExistsError('Validation report already exists; keep previous results and use a new release workspace.')
    for name in ('fastapi', 'httpx', 'uvicorn'):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f'Missing demo test dependency: {name}')
    log = io.StringIO()
    with redirect_stdout(log):
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        test_result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not test_result.wasSuccessful() or test_result.skipped:
        print(log.getvalue())
        raise RuntimeError('Tests failed or skipped; release blocked.')
    files = public_files()
    python_files = [path for path in files if path.suffix == '.py']
    for path in python_files:
        source = path.read_text(encoding='utf-8')
        ast.parse(source, filename=str(path))
        compile(source, str(path), 'exec')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', readme):
        if '://' not in target and not target.startswith('#'):
            # Results reports are produced below.
            if target != 'results/VALIDATION.md' and not (ROOT / target).exists():
                raise FileNotFoundError('Broken README link: ' + target)
    cli_checks = []
    for module in ('prepare_data', 'train_qlora', 'evaluate', 'serve'):
        command = [sys.executable, '-m', 'text2sql_finetune.' + module, '--help']
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if completed.returncode:
            raise RuntimeError('CLI failed: ' + module + '\n' + completed.stderr)
        cli_checks.append({'command': 'python -m text2sql_finetune.' + module + ' --help', 'exit_code': 0})
    demo_spec = importlib.util.spec_from_file_location('local_demo', ROOT / 'examples' / 'demo.py')
    demo = importlib.util.module_from_spec(demo_spec)
    demo_spec.loader.exec_module(demo)
    from fastapi.testclient import TestClient
    from text2sql_finetune.serve import create_app, DEMO_QUESTIONS
    with tempfile.TemporaryDirectory() as directory:
        database = demo.create_demo_database(directory)
        with TestClient(create_app(db_root=directory, demo_mode=True)) as client:
            assert client.get('/').status_code == 200
            assert client.get('/health').json()['mode'] == 'scripted_demo'
            response = client.post('/api/query', json={'database': database.name, 'question': DEMO_QUESTIONS[0]}).json()
            assert response['repair_attempted'] and response['repaired'] and response['error'] is None
            assert len(response['rows']) == 5
    packages = {name: importlib.metadata.version(name) for name in
                ('fastapi', 'starlette', 'uvicorn', 'pydantic', 'pydantic-core', 'httpx',
                 'httpcore', 'anyio', 'h11', 'click', 'typing-extensions', 'typing-inspection',
                 'annotated-types', 'certifi', 'idna')}
    lock = '# Exact dependencies used for local CPU engineering verification.\n'
    lock += '\n'.join(f'{name}=={version}' for name, version in sorted(packages.items())) + '\n'
    (ROOT / 'requirements-demo-lock.txt').write_text(lock, encoding='utf-8')
    local_time = datetime.now(timezone(timedelta(hours=8))).isoformat()
    report = {
        'validation_time_Asia_Shanghai': local_time,
        'scope': 'CPU engineering tests and scripted artificial SQLite demo; not model quality evaluation',
        'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
        'tests': {'command': 'python -m unittest discover -s tests -v', 'total': test_result.testsRun,
                  'failures': len(test_result.failures), 'errors': len(test_result.errors), 'skipped': len(test_result.skipped)},
        'python_syntax_files_checked': len(python_files), 'cli_checks': cli_checks,
        'demo': {'mode': response['mode'], 'notice': response['notice'], 'question': response['question'],
                 'initial_sql': response['initial_sql'], 'initial_error': response['initial_error'],
                 'repaired_sql': response['repaired_sql'], 'repair_attempted': response['repair_attempted'],
                 'repaired': response['repaired'], 'result_rows': response['rows'], 'columns': response['columns']},
        'gpu_training_performed': False, 'spider_model_accuracy_measured': False,
    }
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    (results / 'local_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (results / 'unittest.log').write_text(log.getvalue(), encoding='utf-8')
    (results / 'VALIDATION.md').write_text(
        '# 本地工程验证\n\n'
        f'验证时间（Asia/Shanghai）：{local_time}\n\n'
        f'Python {platform.python_version()}，{test_result.testsRun} 项测试全部通过，无跳过。'
        f'检查 {len(python_files)} 个 Python 文件语法，数据处理、训练、评估和服务 CLI 帮助入口均正常。\n\n'
        '真实调用人工库 API：初始 SQL 的 employee_name 字段不存在，固定脚本修复为 name 后成功返回 5 行。'
        '初始 SQL、SQLite 错误、修复 SQL 和结果保存在 local_validation.json。\n\n'
        '本记录验证工程行为；没有执行 GPU 训练，也没有测量 Qwen/Spider 的准确率。'
        'GitHub Actions 配置已提供，本地通过不表示云端 CI 已运行。\n\n'
        '命令：`python -m unittest discover -s tests -v`；完整记录：`unittest.log`。\n', encoding='utf-8')
    files = public_files() + [results / 'unittest.log']
    files = sorted(set(files))
    names = [path.relative_to(ROOT).as_posix() for path in files]
    names += ['results/release_file_list.txt', 'results/release_hashes.json']
    (results / 'release_file_list.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    files.append(results / 'release_file_list.txt')
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (results / 'release_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    files.append(results / 'release_hashes.json')
    archive = ROOT.parent / 'text2sql-spider-qlora-github.zip'
    if archive.exists():
        raise FileExistsError('Release archive already exists: ' + str(archive))
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
        for path in files:
            output.write(path, 'text2sql-spider-qlora/' + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
        assert not any(any(part in {'.idea', '__pycache__', '.runtime-deps', 'artifacts', 'data'}
                           for part in Path(name).parts) for name in output.namelist())
    print(json.dumps({'tests_passed': test_result.testsRun, 'syntax_checked': len(python_files),
                      'archive_files': len(files), 'archive_bytes': archive.stat().st_size,
                      'archive': str(archive), 'validation': str(results / 'local_validation.json')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
