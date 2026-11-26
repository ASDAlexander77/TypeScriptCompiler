#!/usr/bin/env python3
"""Differential run: execute TypeScript files with tslang (JIT) and with Node, compare the output.

The tsc sweep (scripts/tsc_compiler_tests.ps1) only checks that a file compiles. This script runs
each file and compares stdout and the exit status with Node running the same file, transpiled by
`tsc --noCheck` (TypeScript 7 CLI, no type checking).

Two sources:
  --sweep <dir>      results.csv of a tsc sweep: its passing single-file tests are run
                     (test files are read from --tests-dir)
  --corpus <dir>     a folder of .ts files (e.g. tslang/test/tester/tests); Node gets a prelude
                     with `print` and `assert`, and `main()` is called when the file defines it

Results in --out: results.csv, summary.md, diffs/<name>.txt (both outputs of each mismatch).

Categories:
  match           same stdout, both exited 0 (also: both threw, same stdout before)
  match-silent    match, and neither printed anything (runs, but proves little)
  no-code         tslang found nothing to run (declarations only) and Node printed nothing
  stdout          stdout differs
  tslang-throw    Node exited 0, tslang exited non-zero (uncaught error at run time)
  tslang-crash    tslang died by an exception code (access violation, stack overflow, ...)
  tslang-timeout  tslang ran longer than --timeout, Node did not
  tslang-compile  tslang refused the file (JIT compile error) although the sweep compiled it
  tslang-only-ok  Node threw, tslang exited 0
  node-fail       Node could not run it (transpile or run-time error) and tslang failed too: skipped
"""
import argparse
import csv
import ctypes
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TSLANG = REPO / '__build/tslang/windows-msbuild-2026-release/bin/tslang.exe'

NODE_PRELUDE = r'''
function print(...a) { console.log(...a); }
function assert(c, m) { if (!c) { throw new Error(m === undefined ? "assert" : m); } }
'''
NODE_EPILOGUE = '\nif (typeof main === "function") { main(); }\n'

DIRECTIVE = re.compile(r'^\s*//\s*@(\w+)\s*:', re.I)


def single_unit_source(path: Path):
    """Source of a tsc harness test, or None when it is split into several files."""
    text = path.read_text(encoding='utf-8', errors='replace')
    lines = []
    for line in text.splitlines():
        m = DIRECTIVE.match(line)
        if m:
            if m.group(1).lower() == 'filename':
                return None
            continue
        lines.append(line)
    return '\n'.join(lines) + '\n'


def run(cmd, cwd, timeout, env=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout, env=env)
        return p.returncode, p.stdout.decode('utf-8', 'replace'), p.stderr.decode('utf-8', 'replace'), time.time() - t0
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b'').decode('utf-8', 'replace')
        return 'timeout', out, '', time.time() - t0


def norm(s: str) -> str:
    return '\n'.join(l.rstrip() for l in s.replace('\r\n', '\n').strip().split('\n'))


def is_crash_code(rc):
    # Windows exception codes come back as large unsigned values (0xC0000005 ...), POSIX signals as negatives
    return isinstance(rc, int) and (rc < 0 or rc >= 0xC0000000 or rc in (3, 0x80000003))


def classify(t_rc, t_out, t_err, n_rc, n_out):
    t, n = norm(t_out), norm(n_out)
    if "entry point 'main' not found" in t_err:
        # nothing at the top level to run (declarations only)
        return 'no-code' if not n else 'stdout'
    if n_rc != 0:
        if t_rc == 0:
            return 'tslang-only-ok'
        if t_rc != 'timeout' and not is_crash_code(t_rc) and t == n:
            return 'match'
        return 'node-fail'
    if t_rc == 'timeout':
        return 'tslang-timeout'
    if is_crash_code(t_rc):
        return 'tslang-crash'
    if t_rc != 0:
        if re.search(r': error: ', t_err) and not t:
            return 'tslang-compile'
        return 'tslang-throw'
    if t != n:
        return 'stdout'
    return 'match-silent' if not t else 'match'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', help='folder of a tsc sweep (results.csv)')
    ap.add_argument('--tests-dir', default=r'E:\Dev3\TypeScript\tsc\testdata\tests\cases\compiler')
    ap.add_argument('--corpus', help='folder of .ts files')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tslang', default=str(DEFAULT_TSLANG))
    ap.add_argument('--tslang-args', default='', help='extra tslang arguments, e.g. "--mm=rc"')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=30)
    ap.add_argument('--filter', default='', help='regex on the test name')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    if os.name == 'nt':  # no crash dialogs from crashing children
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    src_dir, js_dir, diff_dir = out / 'src', out / 'js', out / 'diffs'
    for d in (src_dir, js_dir, diff_dir):
        d.mkdir(parents=True)

    # 1. collect the sources
    names = []
    corpus = bool(args.corpus)
    if args.sweep:
        with open(Path(args.sweep) / 'results.csv', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['Status'] == 'pass' and row['Units'] == '1':
                    names.append(row['Test'])
        base = Path(args.tests_dir)
    else:
        base = Path(args.corpus)
        names = sorted(p.name for p in base.glob('*.ts') if not p.name.endswith('.d.ts'))
    if args.filter:
        names = [n for n in names if re.search(args.filter, n)]
    if args.limit:
        names = names[:args.limit]

    tests = []
    for n in names:
        p = base / n
        if p.suffix != '.ts':
            continue
        text = single_unit_source(p) if args.sweep else p.read_text(encoding='utf-8', errors='replace')
        if text is None:
            continue
        stem = p.stem
        (src_dir / (stem + '.ts')).write_text(text, encoding='utf-8')
        tests.append(stem)
    print(f'{len(tests)} tests', flush=True)

    # 2. transpile for Node, one file per tsc call (tsc --noCheck: types stripped, enums/namespaces/parameter
    # properties lowered). A batch call drops the whole batch's output when one file can't be emitted.
    tsc = shutil.which('tsc') or shutil.which('tsc.cmd')

    def transpile(t):
        try:
            subprocess.run([tsc, '--noCheck', '--target', 'es2022', '--module', 'commonjs', '--experimentalDecorators',
                            '--rootDir', str(src_dir), '--outDir', str(js_dir), str(src_dir / (t + '.ts'))],
                           capture_output=True, shell=False, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            return  # tsc 7 hangs on a few inputs; the test then counts as node-fail

        js = js_dir / (t + '.js')
        if corpus and js.exists():
            js.write_text(NODE_PRELUDE + js.read_text(encoding='utf-8') + NODE_EPILOGUE, encoding='utf-8')

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(transpile, tests))
    print(f'{sum(1 for _ in js_dir.glob("*.js"))} transpiled', flush=True)

    env = dict(os.environ)
    for k in ('TSLANG_LIB_PATH', 'DEFAULT_LIB_PATH', 'GC_LIB_PATH', 'LLVM_LIB_PATH'):
        env.setdefault(k, r'I:\tslang')
    extra = args.tslang_args.split()

    # 3. run both
    def one(t):
        ts = src_dir / (t + '.ts')
        js = js_dir / (t + '.js')
        if js.exists():
            n_rc, n_out, n_err, n_s = run(['node', str(js)], js_dir, args.timeout)
        else:
            n_rc, n_out, n_err, n_s = 'no-js', '', 'tsc emitted nothing', 0
        t_rc, t_out, t_err, t_s = run([args.tslang, *extra, str(ts)], src_dir, args.timeout, env)
        cat = classify(t_rc, t_out, t_err, n_rc, n_out)
        if cat not in ('match', 'match-silent', 'node-fail', 'no-code'):
            (diff_dir / (t + '.txt')).write_text(
                f'category: {cat}\ntslang exit: {t_rc}\nnode exit: {n_rc}\n'
                f'===== tslang stdout\n{t_out}\n===== tslang stderr\n{t_err[-4000:]}\n'
                f'===== node stdout\n{n_out}\n===== node stderr\n{n_err[-2000:]}\n', encoding='utf-8')
        first = ''
        if cat == 'stdout':
            a, b = norm(t_out).split('\n'), norm(n_out).split('\n')
            for i in range(max(len(a), len(b))):
                x = a[i] if i < len(a) else '<none>'
                y = b[i] if i < len(b) else '<none>'
                if x != y:
                    first = f'line {i + 1}: tslang={x[:120]!r} node={y[:120]!r}'
                    break
        elif cat in ('tslang-throw', 'tslang-compile', 'tslang-crash'):
            first = next((l for l in t_err.splitlines() if l.strip()), '')[:200]
        return dict(test=t, category=cat, tslang_exit=t_rc, node_exit=n_rc, tslang_s=f'{t_s:.2f}', node_s=f'{n_s:.2f}', first=first)

    started = time.time()
    rows = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for i, r in enumerate(ex.map(one, tests), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f'  {i}/{len(tests)} ({time.time() - started:.0f}s)', flush=True)

    with open(out / 'results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r['category']] = counts.get(r['category'], 0) + 1
    lines = [f'# Differential run: tslang vs Node ({len(rows)} tests)', '',
             f'tslang: `{args.tslang} {args.tslang_args}`'.rstrip(), f'source: `{args.sweep or args.corpus}`', '',
             '| Category | Tests |', '|---|---:|']
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f'| {k} | {v} |')
    for cat in ('tslang-crash', 'tslang-timeout', 'tslang-throw', 'stdout', 'tslang-compile', 'tslang-only-ok'):
        sel = [r for r in rows if r['category'] == cat]
        if sel:
            lines += ['', f'## {cat} ({len(sel)})', '']
            lines += [f'- `{r["test"]}` {r["first"]}'.rstrip() for r in sel]
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:20]))


if __name__ == '__main__':
    main()
