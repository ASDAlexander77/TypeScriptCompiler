#!/usr/bin/env python3
# Differential fuzzer for the memory models. Generates small programs from the shapes that keep
# breaking -mm=own and -mm=rc (aliasing through calls, reassignment inside arguments, views and
# bound methods over a reassigned owner, keep/borrow/return on some paths only), compiles each one
# under own, rc and gc, runs them, and reports the programs where a model disagrees with gc.
# gc is the reference. A program own rejects is fine; a program own accepts must print what gc
# prints. rc must accept and print what gc prints.
#
# Every program ends with a churn loop of same-size allocations before its final reads, and the
# helpers churn before they read too, so a freed block is claimed by something else and a
# use-after-free shows up as a wrong value or as a crash (0xC0000374/0xC0000005, SIGSEGV, SIGABRT).
#
# The options shared by every command go before it: --jobs, --out, --avoid, --recheck-tslang and
# the compiler paths below.
#
#   mm_fuzz.py [--jobs J] [--out DIR] [--avoid SHAPES] [--recheck-tslang LABEL=EXE[,LIB]] \
#              run [--count N] [--seed S]
#       Generate and check N programs. When own rejects a program, the lines own reports are
#       deleted until it accepts (a whole block when the line opens one), and that variant is
#       checked too ("<seed>r"): own accepts few raw programs, and the variants sit right at the
#       edge of what it allows. Each failure is confirmed (the binary is run again), minimized
#       (delta-debugging over lines and blocks, then smaller loop counts) and written to
#       DIR/failures/<name>/ as a standalone repro.ts with asserts on gc's values, beside the
#       original, each model's output and info.json. With --recheck-tslang, each minimized repro
#       is also run with a second compiler (main when fuzzing a branch, or the other way round)
#       and info.json says whether it fails there too. --avoid stops generating a shape whose
#       bug is already known and would hide the rest (arg-assign, globals).
#
#   mm_fuzz.py replay FILE.ts|DIR ...
#       Check existing programs (the seeds/ folder by default), without generating or minimizing.
#       seeds/ holds the probes reviewers wrote by hand against own-raytrace that still failed
#       when this tool was written; the generator's shapes come from them.
#
#   mm_fuzz.py minimize FILE.ts --model own|rc
#       Minimize one failing program.
#
#   mm_fuzz.py report DIR ...
#       Group the minimized failures of earlier runs by model, kind and minimized shape (generated
#       names and numbers normalized) into DIR/report.md.
#
#   mm_fuzz.py show [--seed S]
#       Print one generated program.
#
# The compiler defaults to the build tree beside this checkout (windows-msbuild-2026-release on
# Windows, ninja/release on Linux); --tslang, --tslang-lib-path and --gc-lib-path override it.
# Every compile uses --emit=exe --opt --opt_level=3 --no-default-lib.
#
# On Linux, MALLOC_PERTURB_ is set for the runs: glibc then fills freed memory, so a stale read
# gives a wrong number even when churn did not win the block, and a double free aborts.

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time

IS_WINDOWS = os.name == "nt"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODELS = ("gc", "rc", "own")

COMPILE_TIMEOUT = 120
RUN_TIMEOUT = 30


# ---------------------------------------------------------------------------------------------
# Toolchain


def default_paths():
    if IS_WINDOWS:
        build = os.path.join(REPO, "__build", "tslang", "windows-msbuild-2026-release")
        return (os.path.join(build, "bin", "tslang.exe"), os.path.join(build, "lib"),
                os.path.join(REPO, "3rdParty", "gc", "x64", "release", "lib"))
    build = os.path.join(REPO, "__build", "tslang", "ninja", "release")
    return (os.path.join(build, "bin", "tslang"), os.path.join(build, "lib"),
            os.path.join(REPO, "3rdParty", "gc", "release", "lib"))


class Toolchain:
    def __init__(self, tslang, lib_path, gc_lib_path):
        self.tslang = os.path.abspath(tslang)
        self.lib_path = os.path.abspath(lib_path)
        self.gc_lib_path = os.path.abspath(gc_lib_path)
        for p in (self.tslang, self.lib_path, self.gc_lib_path):
            if not os.path.exists(p):
                sys.exit("FAIL missing prerequisite: " + p)

    def describe(self):
        return self.tslang

    def compile(self, src, exe, mm):
        """Returns (status, code, text). status is ok, rejected, crash or timeout."""
        cmd = [self.tslang, "--emit=exe", "--opt", "--opt_level=3", "--no-default-lib",
               "-mm=" + mm, "--tslang-lib-path=" + self.lib_path,
               "--gc-lib-path=" + self.gc_lib_path, src, "-o", exe]
        try:
            p = subprocess.run(cmd, cwd=os.path.dirname(exe), capture_output=True,
                               timeout=COMPILE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return "timeout", None, ""
        text = (p.stdout + p.stderr).decode("utf-8", "replace")
        if p.returncode == 0 and os.path.exists(exe):
            return "ok", 0, text
        # a diagnostic is a rejection; anything else (no message, an exception code, a link
        # failure) is the compiler's own failure
        if 0 < p.returncode < 256 and "error:" in text and "LNK" not in text and "ld:" not in text:
            return "rejected", p.returncode, text
        return "crash", p.returncode, text


def run_exe(exe):
    """Returns (code, output). code is an int, or 'timeout'."""
    env = dict(os.environ)
    if not IS_WINDOWS:
        env["MALLOC_PERTURB_"] = "165"
    try:
        p = subprocess.run([exe], cwd=os.path.dirname(exe), capture_output=True,
                           timeout=RUN_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace").replace("\r\n", "\n")


def code_name(code):
    if code == "timeout" or code is None:
        return str(code)
    if IS_WINDOWS and code >= 0x80000000:
        return "0x%08X" % code
    if code < 0:
        return "SIG%d" % -code
    return str(code)


# ---------------------------------------------------------------------------------------------
# Checking one program


class Result:
    """What one model did with one program."""

    def __init__(self, mm, status, code=None, text="", run_code=None, output=""):
        self.mm = mm
        self.status = status          # ok, rejected, crash, timeout (compile)
        self.code = code
        self.text = text              # compiler output
        self.run_code = run_code      # exit code or 'timeout' when compiled
        self.output = output

    def summary(self):
        if self.status != "ok":
            return "compile-" + self.status + ("" if self.code is None else ":" + code_name(self.code))
        if self.run_code != 0:
            return "run:" + code_name(self.run_code)
        return "ok"


def ext():
    return ".exe" if IS_WINDOWS else ""


def build_and_run(tc, src, work, mm):
    os.makedirs(work, exist_ok=True)
    exe = os.path.join(work, "p_" + mm + ext())
    if os.path.exists(exe):
        os.remove(exe)
    status, code, text = tc.compile(src, exe, mm)
    if status != "ok":
        return Result(mm, status, code, text)
    rc, out = run_exe(exe)
    return Result(mm, "ok", 0, text, rc, out)


def verdict(gc, r):
    """Classifies model result r against the gc result. Returns None when they agree (or own
    rejected), else a failure kind: crash, wrong, timeout, rejects (rc only), compiler-crash."""
    if r.status == "rejected":
        return "rejects" if r.mm == "rc" else None
    if r.status in ("crash", "timeout"):
        return "compiler-" + r.status
    if r.run_code == "timeout":
        return "timeout"
    if r.run_code != 0:
        return "crash"
    if r.output != gc.output:
        return "wrong"
    return None


def check_program(tc, text, work, models=("rc", "own")):
    """Writes text to work/p.ts and runs it under gc and the given models.
    Returns (gc_result, {mm: result}, {mm: kind})."""
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "p.ts")
    with open(src, "w", newline="\n") as f:
        f.write(text)
    gc = build_and_run(tc, src, work, "gc")
    results, kinds = {}, {}
    if gc.status != "ok" or gc.run_code != 0:
        return gc, results, kinds
    for mm in models:
        r = build_and_run(tc, src, work, mm)
        results[mm] = r
        k = verdict(gc, r)
        if k:
            kinds[mm] = k
    return gc, results, kinds


# ---------------------------------------------------------------------------------------------
# The generator
#
# The shapes come from the probes reviewers wrote by hand against own-raytrace (seeds/): a class
# with an object field and an array field, @noinline factories so stack promotion cannot hide a
# free, a churn() that refills freed blocks, helpers that keep, borrow or return a parameter on
# some paths, and a main loop that aliases, reassigns and passes the same object twice.

HEADER = [
    "class W { x: number; constructor(x: number) { this.x = x; } }",
    "class V { x: number; w: W; a: number[]; constructor(x: number) { this.x = x; this.w = new W(x); this.a = [x]; } get(): number { return this.x; } }",
    "class N { v: V; items: V[]; constructor(v: V) { this.items = [new V(v.x)]; this.v = v; } }",
    "class H { n: N = new N(new V(0)); }",
    "interface IX { v: V; }",
    "interface IY { v: V; items: V[]; }",
    "interface IG { get(): number; }",
    "@noinline function mk(i: number): N { return new N(new V(i)); }",
    "@noinline function mv(i: number): V { return new V(i); }",
    "function churn() { const hold: N[] = []; for (let i = 0; i < 64; i++) { hold.push(new N(new V(777))); } }",
]

AVOIDABLE = {
    "arg-assign": "an assignment inside an argument list, f(n, (n = mk(1), m))",
    "globals": "the module-level G, HG, GS and GA",
}

TYPE_NAME = {"N": "N", "V": "V", "NA": "N[]", "IX": "IX", "IY": "IY", "IG": "IG",
             "FN": "() => number", "SV": "Shared<V>", "H": "H"}

# number-valued reads of a value of each type
READS = {
    "N": ["{0}.v.x", "{0}.items[0].x", "{0}.v.w.x", "{0}.v.a[0]", "{0}.v.get()"],
    "V": ["{0}.x", "{0}.w.x", "{0}.a[0]", "{0}.get()"],
    "NA": ["{0}[0].v.x", "{0}[{0}.length - 1].v.x", "{0}[0].items[0].x"],
    "IX": ["{0}.v.x", "{0}.v.w.x"],
    "IY": ["{0}.v.x", "{0}.items[0].x"],
    "IG": ["{0}.get()"],
    "FN": ["{0}()"],
    "SV": ["{0}.value.x", "{0}.value.w.x"],
    "H": ["{0}.n.v.x", "{0}.n.items[0].x"],
}

PREFIX = {"N": "n", "V": "v", "NA": "a", "IX": "ix", "IY": "iy", "IG": "ig", "FN": "c",
          "SV": "sh", "H": "h"}


class Var:
    def __init__(self, name, ty, mutable, depth, minlen=0):
        self.name = name
        self.ty = ty
        self.mutable = mutable
        self.depth = depth        # block nesting where it was declared
        self.minlen = minlen      # arrays: the length it has at least, in that block


class Helper:
    def __init__(self, name, params, ret):
        self.name = name
        self.params = params      # list of types; 'B' is a boolean
        self.ret = ret            # 'num' or 'N'


class Ctx:
    def __init__(self, gen, fn, acc):
        self.gen = gen
        self.fn = fn              # 'main', 'helper', 'helperN'
        self.acc = acc
        self.scopes = [[]]
        self.loop_depth = 0
        self.loop_vars = []
        self.in_try = 0

    def vars(self, pred=lambda v: True):
        out = [v for v in self.gen.globals if pred(v)]
        for s in self.scopes:
            out.extend(v for v in s if pred(v))
        return out

    def declare(self, ty, mutable, minlen=0):
        v = Var(self.gen.fresh(PREFIX[ty]), ty, mutable, len(self.scopes), minlen)
        self.scopes[-1].append(v)
        return v


class Gen:
    def __init__(self, seed, size=None, avoid=()):
        self.r = random.Random(seed)
        self.seed = seed
        self.avoid = set(avoid)   # shapes not to generate, once their bugs are known (AVOIDABLE)
        self.counter = 0
        self.globals = []
        self.helpers = []
        self.reducers = []
        self.size = size if size is not None else self.r.choice([1, 2, 2, 3])

    # -- small pieces

    def fresh(self, prefix):
        self.counter += 1
        return "%s%d" % (prefix, self.counter)

    def k(self):
        return self.r.randint(1, 9)

    def chance(self, p):
        return self.r.random() < p

    def pick(self, seq):
        return self.r.choice(seq)

    def num_k(self, ctx):
        if ctx.loop_vars and self.chance(0.5):
            return "%s + %d" % (self.pick(ctx.loop_vars), self.k())
        return str(self.k())

    def cond(self, ctx):
        if ctx.loop_vars and self.chance(0.6):
            return "%s %% %d == %d" % (self.pick(ctx.loop_vars), self.r.randint(2, 3), self.r.randint(0, 1))
        bools = ctx.vars(lambda v: v.ty == "B")
        if bools and self.chance(0.5):
            return self.pick(bools).name
        return "%s > %d" % (self.num_read(ctx), self.k())

    def num_read(self, ctx, no_calls=False):
        # a closure's body reads no function value: `c = () => c()` would recurse forever
        vs = ctx.vars(lambda v: v.ty in READS and not (no_calls and v.ty == "FN"))
        if not vs:
            return str(self.k())
        v = self.pick(vs)
        return self.pick(READS[v.ty]).format(v.name)

    # -- expressions of a type

    def expr(self, ctx, ty, fresh_bias=0.4, depth=0):
        vs = ctx.vars(lambda v: v.ty == ty)
        alias = vs and not self.chance(fresh_bias)
        if ty == "N":
            opts = []
            if alias:
                opts.append(lambda: self.pick(vs).name)
            hs = ctx.vars(lambda v: v.ty == "H")
            if hs:
                opts.append(lambda: self.pick(hs).name + ".n")
            arrs = ctx.vars(lambda v: v.ty == "NA")
            if arrs:
                opts.append(lambda: self.pick(arrs).name + "[0]")
            if depth == 0:
                rets = [h for h in self.helpers if h.ret == "N"]
                if rets and self.chance(0.3):
                    opts.append(lambda: self.call(ctx, self.pick(rets), depth + 1))
                if arrs and self.reducers and self.chance(0.4):
                    opts.append(lambda: "%s.reduce(%s, %s)" % (self.pick(arrs).name, self.pick(self.reducers),
                                                               self.expr(ctx, "N", 0.7, depth + 1)))
            if not opts or self.chance(fresh_bias):
                return "mk(%s)" % self.num_k(ctx)
            return self.pick(opts)()
        if ty == "V":
            opts = []
            if alias:
                opts.append(lambda: self.pick(vs).name)
            ns = ctx.vars(lambda v: v.ty in ("N", "IX", "IY"))
            if ns:
                opts.append(lambda: self.pick(ns).name + ".v")
            ns2 = ctx.vars(lambda v: v.ty in ("N", "IY"))
            if ns2:
                opts.append(lambda: self.pick(ns2).name + ".items[0]")
            ss = ctx.vars(lambda v: v.ty == "SV")
            if ss:
                opts.append(lambda: self.pick(ss).name + ".value")
            if not opts or self.chance(fresh_bias):
                return "mv(%s)" % self.num_k(ctx)
            return self.pick(opts)()
        if ty == "NA":
            if alias and self.chance(0.5):
                return self.pick(vs).name
            return "[" + ", ".join(self.expr(ctx, "N", 0.6, depth + 1)
                                   for _ in range(self.r.randint(1, 3))) + "]"
        if ty == "IX":
            opts = [lambda: self.expr(ctx, "N", fresh_bias, depth + 1),
                    lambda: "{ v: %s }" % self.expr(ctx, "V", 0.6, depth + 1)]
            if alias:
                opts.append(lambda: self.pick(vs).name)
            iys = ctx.vars(lambda v: v.ty == "IY")
            if iys:
                opts.append(lambda: self.pick(iys).name)
            return self.pick(opts)()
        if ty == "IY":
            opts = [lambda: self.expr(ctx, "N", fresh_bias, depth + 1),
                    lambda: "{ v: %s, items: [%s] }" % (self.expr(ctx, "V", 0.6, depth + 1),
                                                       self.expr(ctx, "V", 0.8, depth + 1))]
            if alias:
                opts.append(lambda: self.pick(vs).name)
            return self.pick(opts)()
        if ty == "IG":
            if alias and self.chance(0.5):
                return self.pick(vs).name
            return self.expr(ctx, "V", fresh_bias, depth + 1)
        if ty == "FN":
            if alias and self.chance(0.3):
                return self.pick(vs).name
            c = self.r.random()
            if c < 0.4:
                return "(): number => %s" % self.num_read(ctx, no_calls=True)
            if c < 0.55:
                assignable = ctx.vars(lambda v: v.ty == "N" and v.mutable)
                if assignable:
                    v = self.pick(assignable)
                    return "(): number => { %s = mk(%d); return %s; }" % (
                        v.name, self.k(), self.pick(READS["N"]).format(v.name))
            igs = ctx.vars(lambda v: v.ty == "IG")
            if igs and self.chance(0.3):
                return self.pick(igs).name + ".get"
            return self.expr(ctx, "V", fresh_bias, depth + 1) + ".get"
        if ty == "SV":
            if alias and self.chance(0.6):
                return self.pick(vs).name
            return "new Shared<V>(%s)" % self.expr(ctx, "V", 0.7, depth + 1)
        if ty == "H":
            if alias:
                return self.pick(vs).name
            return "new H()"
        if ty == "B":
            return self.cond(ctx)
        raise ValueError(ty)

    def arg(self, ctx, ty, previous):
        """An argument of type ty. Biased toward the shapes that alias: the same object twice,
        a field of an earlier argument, and an assignment inside the argument list. previous
        holds the (type, expression) of the arguments before it."""
        c = self.r.random()
        names = [(t, e) for t, e in previous if re.fullmatch(r"[A-Za-z_]\w*", e)]
        if ty == "N" and c < 0.3:
            same = [e for t, e in names if t == "N"]
            if same:
                return self.pick(same)
        if ty == "V" and c < 0.3:
            owners = [e for t, e in names if t in ("N", "IX", "IY")]
            if owners:
                return self.pick(owners) + ".v"
        if c < 0.55 and "arg-assign" not in self.avoid:
            assignable = ctx.vars(lambda v: v.ty == ty and v.mutable)
            hs = ctx.vars(lambda v: v.ty == "H")
            if assignable and self.chance(0.7):
                v = self.pick(assignable)
                return "(%s = %s, %s)" % (v.name, self.expr(ctx, ty, 0.9, 1), self.expr(ctx, ty, 0.2, 1))
            if ty == "N" and hs:
                h = self.pick(hs)
                return "(%s.n = mk(%d), %s)" % (h.name, self.k(), self.expr(ctx, ty, 0.2, 1))
        return self.expr(ctx, ty, 0.25, 1)

    def call(self, ctx, h, depth=0):
        args = []
        for ty in h.params:
            args.append((ty, self.cond(ctx) if ty == "B" else self.arg(ctx, ty, args)))
        return "%s(%s)" % (h.name, ", ".join(e for _, e in args))

    # -- statements

    def stmts(self, ctx, count, ind):
        out = []
        for _ in range(count):
            out.extend(self.stmt(ctx, ind))
        return out

    def block(self, ctx, count, ind, loop_var=None):
        ctx.scopes.append([])
        if loop_var:
            ctx.loop_vars.append(loop_var)
        body = self.stmts(ctx, count, ind)
        if loop_var:
            ctx.loop_vars.pop()
        ctx.scopes.pop()
        return body

    def stmt(self, ctx, ind, depth=0):
        sp = "    " * ind
        kinds = [("decl", 6), ("assign", 5), ("field", 4), ("call", 6), ("read", 4), ("churn", 2)]
        if ind < 4:
            kinds += [("if", 2), ("for", 2), ("try", 1)]
        if ctx.loop_depth > 0:
            kinds.append(("jump", 2))
        if ctx.fn != "main":
            kinds.append(("return", 1))
        kind = self.r.choices([k for k, _ in kinds], [w for _, w in kinds])[0]
        acc = ctx.acc

        if kind == "decl":
            ty = self.r.choices(["N", "V", "NA", "IX", "IY", "IG", "FN", "SV"],
                                [6, 4, 3, 2, 1, 1, 3, 1])[0]
            mutable = self.chance(0.6)
            e = self.expr(ctx, ty)
            v = ctx.declare(ty, mutable, 1 if ty == "NA" else 0)
            annotate = mutable or ty in ("IX", "IY", "IG", "NA", "SV") or self.chance(0.3)
            if ty == "FN" and not mutable and self.chance(0.6):
                annotate = False
            return ["%s%s %s%s = %s;" % (sp, "let" if mutable else "const", v.name,
                                         ": " + TYPE_NAME[ty] if annotate else "", e)]

        if kind == "assign":
            vs = ctx.vars(lambda v: v.mutable and v.ty in TYPE_NAME and v.ty != "H")
            if not vs:
                return self.stmt(ctx, ind)
            v = self.pick(vs)
            e = self.expr(ctx, v.ty, 0.5)
            if v.ty == "NA":
                v.minlen = 1
            return ["%s%s = %s;" % (sp, v.name, e)]

        if kind == "field":
            opts = []
            for v in ctx.vars(lambda v: v.ty in ("N", "IX", "IY")):
                opts.append("%s.v = %s;" % (v.name, self.expr(ctx, "V", 0.6)))
            for v in ctx.vars(lambda v: v.ty in ("N", "IY")):
                opts.append("%s.items[0] = %s;" % (v.name, self.expr(ctx, "V", 0.6)))
                opts.append("%s.items.push(%s);" % (v.name, self.expr(ctx, "V", 0.4)))
            for v in ctx.vars(lambda v: v.ty == "H"):
                opts.append("%s.n = %s;" % (v.name, self.expr(ctx, "N", 0.5)))
            for v in ctx.vars(lambda v: v.ty == "V"):
                opts.append("%s.w = new W(%d);" % (v.name, self.k()))
            for v in ctx.vars(lambda v: v.ty == "SV"):
                opts.append("%s.value = %s;" % (v.name, self.expr(ctx, "V", 0.6)))
            for v in ctx.vars(lambda v: v.ty == "NA"):
                opts.append("%s[0] = %s;" % (v.name, self.expr(ctx, "N", 0.5)))
                opts.append("%s.push(%s);" % (v.name, self.expr(ctx, "N", 0.4)))
                if v.minlen >= 2 and v.depth == len(ctx.scopes):
                    opts.append("%s.pop();" % v.name)
            if not opts:
                return self.stmt(ctx, ind)
            s = self.pick(opts)
            m = re.match(r"(\w+)\.(push|pop)\(", s)
            if m:
                for v in ctx.vars(lambda v: v.name == m.group(1) and v.ty == "NA"):
                    if v.depth == len(ctx.scopes):
                        v.minlen += 1 if m.group(2) == "push" else -1
            return [sp + s]

        if kind == "call":
            if not self.helpers:
                return self.stmt(ctx, ind)
            h = self.pick(self.helpers)
            if h.ret == "num":
                return ["%s%s += %s;" % (sp, acc, self.call(ctx, h))]
            target = ctx.vars(lambda v: v.ty == "N" and v.mutable)
            if target and self.chance(0.5):
                return ["%s%s = %s;" % (sp, self.pick(target).name, self.call(ctx, h))]
            mutable = self.chance(0.5)
            c = self.call(ctx, h)
            v = ctx.declare("N", mutable)
            return ["%s%s %s: N = %s;" % (sp, "let" if mutable else "const", v.name, c)]

        if kind == "read":
            if self.chance(0.15):
                arrs = ctx.vars(lambda v: v.ty == "NA")
                if arrs:
                    return ["%s%s += %s.reduce((t: number, x: N) => t + x.v.x, 0);" % (
                        sp, acc, self.pick(arrs).name)]
            return ["%s%s += %s;" % (sp, acc, self.num_read(ctx))]

        if kind == "churn":
            return [sp + "churn();"]

        if kind == "if":
            c = self.cond(ctx)
            then = self.block(ctx, self.r.randint(1, 3), ind + 1)
            if self.chance(0.5):
                other = self.block(ctx, self.r.randint(1, 2), ind + 1)
                return ["%sif (%s) {" % (sp, c)] + then + [sp + "} else {"] + other + [sp + "}"]
            return ["%sif (%s) {" % (sp, c)] + then + [sp + "}"]

        if kind == "for":
            j = self.fresh("j")
            ctx.loop_depth += 1
            body = self.block(ctx, self.r.randint(1, 4), ind + 1, j)
            ctx.loop_depth -= 1
            return ["%sfor (let %s = 0; %s < %d; %s++) {" % (sp, j, j, self.r.randint(2, 4), j)] + body + [sp + "}"]

        if kind == "try":
            ctx.in_try += 1
            body = self.block(ctx, self.r.randint(1, 3), ind + 1)
            ctx.in_try -= 1
            fin = self.block(ctx, self.r.randint(1, 2), ind + 1)
            return [sp + "try {"] + body + [sp + "} finally {"] + fin + [sp + "}"]

        if kind == "jump":
            return ["%sif (%s) { %s; }" % (sp, self.cond(ctx), self.pick(["continue", "break"]))]

        if kind == "return":
            if ctx.fn == "helper":
                return ["%sif (%s) { return %s + %s; }" % (sp, self.cond(ctx), acc, self.num_read(ctx))]
            return ["%sif (%s) { return %s; }" % (sp, self.cond(ctx), self.expr(ctx, "N", 0.3))]

        raise ValueError(kind)

    # -- functions

    def helper_params(self):
        shapes = [["N", "N"], ["N", "V"], ["NA", "N"], ["H", "N"], ["IX", "N"], ["N", "N", "B"],
                  ["N"], ["SV", "N"], ["N", "IY"]]
        return self.pick(shapes)

    def gen_helper(self):
        params = self.helper_params()
        ret = "N" if self.chance(0.35) and "N" in params else "num"
        name = self.fresh("g" if ret == "N" else "f")
        ctx = Ctx(self, "helperN" if ret == "N" else "helper", "acc")
        names = []
        for ty in params:
            pn = self.fresh("p")
            names.append("%s: %s" % (pn, "boolean" if ty == "B" else TYPE_NAME[ty]))
            ctx.scopes[0].append(Var(pn, ty, True, 1, 1 if ty == "NA" else 0))
        lines = []
        if self.chance(0.7):
            lines.append("@noinline")
        lines.append("function %s(%s): %s {" % (name, ", ".join(names), "N" if ret == "N" else "number"))
        if ret == "num":
            lines.append("    let acc: number = 0;")
        else:
            lines.append("    let acc: number = 0;")
        body = self.stmts(ctx, self.r.randint(1, 2 + self.size), 1)
        lines += body
        if self.chance(0.6):
            lines.append("    churn();")
        if ret == "num":
            lines.append("    return acc + %s;" % self.num_read(ctx))
        else:
            ns = [v for v in ctx.scopes[0] if v.ty == "N"]
            if self.chance(0.7) and ns:
                lines.append("    return %s;" % self.pick(ns).name)
            else:
                lines.append("    return %s;" % self.expr(ctx, "N", 0.3))
        lines.append("}")
        self.helpers.append(Helper(name, params, ret))
        return lines

    def gen_reducer(self):
        name = self.fresh("r")
        k = self.k()
        body = self.pick([
            "return x.v.x > %d ? acc : x;" % k,
            "return x.v.x > %d ? x : acc;" % k,
            "return x.v.x > %d ? acc : mk(acc.v.x + x.v.x);" % k,
            "return mk(acc.v.x + 1);",
            "if (x.v.x > %d) { return acc; } churn(); return x;" % k,
        ])
        self.reducers.append(name)
        return ["function %s(acc: N, x: N): N { %s }" % (name, body)]

    def gen_globals(self):
        lines = []
        if "globals" in self.avoid:
            return lines
        if self.chance(0.5):
            v = Var("G", "N", True, 0)
            self.globals.append(v)
            lines.append("let G: N = mk(%d);" % self.k())
        if self.chance(0.4):
            self.globals.append(Var("HG", "H", False, 0))
            lines.append("const HG = new H();")
        if self.chance(0.15):
            self.globals.append(Var("GS", "SV", True, 0))
            lines.append("let GS: Shared<V> = new Shared<V>(mv(%d));" % self.k())
        if self.chance(0.2):
            self.globals.append(Var("GA", "NA", False, 0, 1))
            lines.append("const GA: N[] = [mk(%d)];" % self.k())
        return lines

    def program(self):
        lines = list(HEADER)
        lines += self.gen_globals()
        if self.chance(0.5):
            lines += self.gen_reducer()
        for _ in range(self.r.randint(1, 1 + self.size // 2 + 1)):
            lines += self.gen_helper()
        ctx = Ctx(self, "main", "s")
        main = ["function main() {", "    let s: number = 0;"]
        main += self.stmts(ctx, self.r.randint(1, 2 + self.size // 2), 1)
        iters = self.pick([1, 3, 20, 100, 300])
        ctx.loop_vars.append("i")
        ctx.loop_depth += 1
        body = self.block(ctx, self.r.randint(2, 3 + 2 * self.size), 2)
        ctx.loop_depth -= 1
        ctx.loop_vars.pop()
        main += ["    for (let i = 0; i < %d; i++) {" % iters] + body + ["    }"]
        main.append("    churn();")
        checks = ["s"]
        for v in ctx.vars(lambda v: v.ty in READS):
            checks.append(self.pick(READS[v.ty]).format(v.name))
        for n, c in enumerate(checks):
            main.append('    print("c%d", %s);' % (n, c))
        main.append("}")
        return "\n".join(lines + main) + "\n"


# ---------------------------------------------------------------------------------------------
# Asserts on gc's values, for the standalone repro

CHECK_RE = re.compile(r'^(\s*)print\("(c\d+)", (.*)\);\s*$')


def add_asserts(text, gc_output):
    values = {}
    for line in gc_output.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and re.fullmatch(r"c\d+", parts[0]):
            values[parts[0]] = parts[1].strip()
    out = []
    for line in text.splitlines():
        out.append(line)
        m = CHECK_RE.match(line)
        if m and m.group(2) in values and re.fullmatch(r"-?\d+(\.\d+)?", values[m.group(2)]):
            out.append('%sassert(%s == %s, "%s");' % (m.group(1), m.group(3), values[m.group(2)], m.group(2)))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
# Minimization: delta debugging over lines, then over blocks (drop a block, or keep its body
# without the braces), then smaller loop counts


def block_spans(lines):
    """(open, close) line index pairs of the blocks that open at the end of a line."""
    spans, stack = [], []
    for i, line in enumerate(lines):
        s = line.strip()
        opens = s.endswith("{")
        if s.startswith("}") and stack:
            o = stack.pop()
            spans.append((o, i))
        if opens:
            stack.append(i)
    return spans


ERROR_LINE_RE = re.compile(r"\.ts:(\d+):\d+: error:")


def own_repair(tc, text, work, rounds=10):
    """Deletes the lines -mm=own reports errors on (a whole block when the line opens one) until
    own accepts the program. Returns the accepted program, or None."""
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "repair.ts")
    exe = os.path.join(work, "repair" + ext())
    lines = text.rstrip("\n").splitlines()
    for _ in range(rounds):
        with open(src, "w", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        status, code, out = tc.compile(src, exe, "own")
        if status == "ok":
            return "\n".join(lines) + "\n"
        if status != "rejected":
            return None
        bad = sorted({int(m.group(1)) - 1 for m in ERROR_LINE_RE.finditer(out)})
        bad = [b for b in bad if 0 <= b < len(lines)]
        if not bad:
            return None
        spans = {o: c for o, c in block_spans(lines)}
        drop, replace = set(), {}
        for b in bad:
            s = lines[b].strip()
            if s.startswith("}") and s.endswith("{"):
                # `} else {`: drop the else branch, keep the brace that closes the then branch
                c = spans.get(b, b)
                drop.update(range(b, c + 1))
                replace[b] = lines[b][:len(lines[b]) - len(lines[b].lstrip())] + "}"
            elif s.endswith("{") and b in spans:
                drop.update(range(b, spans[b] + 1))
            elif s == "@noinline" or s.startswith("}"):
                return None
            elif s.startswith("return ") and lines[b].startswith("    return "):
                # a function's last return: give it a fresh value instead of no return at all
                fn = next((l for l in reversed(lines[:b]) if l.startswith("function ")), "")
                fallback = "    return mk(1);" if fn.endswith("): N {") else "    return acc;"
                if lines[b] == fallback:
                    return None
                replace[b] = fallback
                drop.add(b)
            else:
                drop.add(b)
        lines = [replace.get(i, l) for i, l in enumerate(lines) if i not in drop or i in replace]
        # a function whose @noinline line lost its function
        lines = [l for i, l in enumerate(lines)
                 if not (l.strip() == "@noinline" and (i + 1 >= len(lines) or not lines[i + 1].startswith("function")))]
    return None


def well_formed(lines):
    """tslang accepts a typed function that falls off its end; keep the minimizer from making one,
    since what it then returns is undefined and the repro would be about that."""
    spans = dict(block_spans(lines))
    for i, line in enumerate(lines):
        if line.startswith("function ") and line.endswith(("): N {", "): number {")) and i in spans:
            if not any(l.startswith("    return ") for l in lines[i + 1:spans[i]]):
                return False
    return True


class Minimizer:
    def __init__(self, tc, model, kind, work):
        self.tc = tc
        self.model = model
        self.kind = kind
        self.work = work
        self.tries = 0
        self.cache = {}

    def interesting(self, lines):
        if not well_formed(lines):
            return False
        text = "\n".join(lines) + "\n"
        key = hashlib.sha1(text.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        self.tries += 1
        gc, results, kinds = check_program(self.tc, text, os.path.join(self.work, "m"), (self.model,))
        ok = kinds.get(self.model) == self.kind
        self.cache[key] = ok
        return ok

    def ddmin(self, lines):
        n = 2
        while len(lines) >= 2:
            chunk = max(1, len(lines) // n)
            removed = False
            start = 0
            while start < len(lines):
                cand = lines[:start] + lines[start + chunk:]
                if cand and self.interesting(cand):
                    lines = cand
                    n = max(n - 1, 2)
                    removed = True
                else:
                    start += chunk
            if not removed:
                if chunk == 1:
                    break
                n = min(n * 2, len(lines))
        return lines

    def blocks(self, lines):
        changed = True
        while changed:
            changed = False
            for o, c in sorted(block_spans(lines), key=lambda s: s[0] - s[1]):
                for cand in (lines[:o] + lines[c + 1:],
                             lines[:o] + [l[4:] if l.startswith("    ") else l for l in lines[o + 1:c]] + lines[c + 1:]):
                    if self.interesting(cand):
                        lines = cand
                        changed = True
                        break
                if changed:
                    break
        return lines

    def loops(self, lines):
        for i, line in enumerate(lines):
            m = re.search(r"< (\d+);", line)
            if m and line.lstrip().startswith("for (") and "churn" not in line:
                for smaller in (1, 2, 3, 20):
                    if smaller < int(m.group(1)):
                        cand = list(lines)
                        cand[i] = line[:m.start(1)] + str(smaller) + line[m.end(1):]
                        if self.interesting(cand):
                            lines = cand
                            break
        return lines

    def run(self, text):
        lines = text.rstrip("\n").splitlines()
        if not self.interesting(lines):
            return None
        prev = None
        while prev != lines:
            prev = lines
            lines = self.ddmin(lines)
            lines = self.blocks(lines)
            lines = self.loops(lines)
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Driver


def short(text, limit=4000):
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]\n"


def confirm(tc, work, mm, kind, times=3):
    """Runs the failing binary again; returns how many of the runs failed the same way."""
    exe = os.path.join(work, "p_" + mm + ext())
    gc_out = open(os.path.join(work, "gc.out")).read() if os.path.exists(os.path.join(work, "gc.out")) else None
    n = 0
    for _ in range(times):
        code, out = run_exe(exe)
        if kind == "crash" and code not in (0, "timeout"):
            n += 1
        elif kind == "wrong" and code == 0 and out != gc_out:
            n += 1
        elif kind == "timeout" and code == "timeout":
            n += 1
        elif kind not in ("crash", "wrong", "timeout"):
            n += 1
    return n


def write_failure(dest, name, original, repro, gc, res, extra):
    d = os.path.join(dest, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "original.ts"), "w", newline="\n") as f:
        f.write(original)
    with open(os.path.join(d, "repro.ts"), "w", newline="\n") as f:
        f.write(repro)
    for mm, r in [("gc", gc)] + sorted(res.items()):
        with open(os.path.join(d, mm + ".out"), "w", newline="\n") as f:
            f.write("# %s\n" % r.summary())
            if r.status != "ok":
                f.write(short(r.text))
            else:
                f.write(short(r.output))
    with open(os.path.join(d, "info.json"), "w", newline="\n") as f:
        json.dump(extra, f, indent=2)
    return d


def process_failure(tc, recheck, text, work, model, kind, dest, name, do_minimize):
    info = {"model": model, "kind": kind, "tslang": tc.describe()}
    if kind in ("crash", "wrong", "timeout"):
        info["confirmed"] = "%d/3" % confirm(tc, work, model, kind)
    repro = text
    if do_minimize and kind in ("crash", "wrong", "rejects", "timeout"):
        m = Minimizer(tc, model, kind, os.path.join(work, "min"))
        t0 = time.time()
        small = m.run(text)
        info["minimize_tries"] = m.tries
        info["minimize_seconds"] = round(time.time() - t0, 1)
        if small:
            repro = small
    info["shape"] = shape_key(repro)
    info["lines"] = len(repro.splitlines())
    gc, res, kinds = check_program(tc, repro, os.path.join(work, "final"), ("rc", "own"))
    if gc.status == "ok" and gc.run_code == 0:
        # the standalone repro asserts gc's values; it must still pass under gc, and the outputs
        # written beside it are the ones of the asserted program
        asserted = add_asserts(repro, gc.output)
        g2, r2, k2 = check_program(tc, asserted, os.path.join(work, "final2"), ("rc", "own"))
        if g2.status == "ok" and g2.run_code == 0:
            repro, gc, res, kinds = asserted, g2, r2, k2
        else:
            info["asserted_gc"] = g2.summary()
    info["final_kinds"] = kinds
    info["final"] = {mm: r.summary() for mm, r in res.items()}
    info["final"]["gc"] = gc.summary()
    for label, other in recheck:
        # any failure counts: with asserts a wrong value is an assert failure there
        g3, r3, k3 = check_program(other, repro, os.path.join(work, "recheck-" + label), (model,))
        info["recheck_" + label] = {"tslang": other.describe(), "gc": g3.summary(),
                                    model: r3[model].summary() if model in r3 else None,
                                    "kind": k3.get(model), "fails": model in k3}
    write_failure(dest, name, text, repro, gc, res, info)
    return info


def kinds_match(k, kind):
    return k is not None and k == kind


GENERATED_NAME_RE = re.compile(r"\b(n|v|a|ix|iy|ig|c|sh|h|p|f|g|r|j)\d+\b")


def shape_key(text):
    """A hash of a minimized repro with generated names and numbers normalized, so that the
    same minimized shape found from different seeds groups together."""
    names = {}

    def name(m):
        return names.setdefault(m.group(0), "%s#%d" % (m.group(1), len(names)))

    t = GENERATED_NAME_RE.sub(name, text)
    t = re.sub(r"\b\d+\b", "K", t)
    return hashlib.sha1(t.encode()).hexdigest()[:10]


def cmd_report(args, tc, recheck):
    """Groups the failures under the given output folders by model, kind and minimized shape."""
    groups = {}
    for root in args.dirs:
        fdir = os.path.join(root, "failures")
        if not os.path.isdir(fdir):
            continue
        for name in sorted(os.listdir(fdir)):
            p = os.path.join(fdir, name, "info.json")
            if not os.path.exists(p):
                continue
            info = json.load(open(p))
            if "shape" not in info or "minimize_tries" not in info:
                continue
            key = (info["model"], info["kind"], info["shape"])
            groups.setdefault(key, []).append((os.path.join(fdir, name), info))
    out = ["# mm_fuzz report", ""]
    out.append("| model | kind | shape | count | lines | example | rechecks |")
    out.append("|---|---|---|---|---|---|---|")
    for (model, kind, shape), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -len(kv[1]))):
        d, info = min(items, key=lambda it: it[1].get("lines", 999))
        rech = ", ".join("%s:%s" % (k[len("recheck_"):], "fails" if v.get("fails") else "passes")
                         for k, v in sorted(info.items()) if k.startswith("recheck_"))
        out.append("| %s | %s | %s | %d | %s | %s | %s |" % (model, kind, shape, len(items), info.get("lines"),
                                                         d, rech))
    out.append("")
    for (model, kind, shape), items in sorted(groups.items()):
        d, info = min(items, key=lambda it: it[1].get("lines", 999))
        out += ["## %s %s %s (%d)" % (model, kind, shape, len(items)), "", "```ts",
                open(os.path.join(d, "repro.ts")).read().rstrip("\n"), "```", ""]
        for mm in ("gc", "rc", "own"):
            p = os.path.join(d, mm + ".out")
            if os.path.exists(p):
                out += ["%s: `%s`" % (mm, open(p).read().splitlines()[0][2:])]
        out.append("")
    text = "\n".join(out) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", newline="\n") as f:
        f.write(text)
    print(text[:6000])
    return 0


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = {}

    def add(self, key):
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1


def cmd_run(args, tc, recheck):
    out = os.path.abspath(args.out)
    work_root = os.path.join(out, "work")
    os.makedirs(work_root, exist_ok=True)
    fail_dir = os.path.join(out, "failures")
    invalid_dir = os.path.join(out, "invalid")
    stats = Stats()
    tsv_lock = threading.Lock()
    tsv = open(os.path.join(out, "results.tsv"), "a", newline="\n")
    minimized_per_sig = {}
    sig_lock = threading.Lock()
    failures = []

    def variant(label, text, work):
        """Checks one program; returns True when own rejected it."""
        gc, res, kinds = check_program(tc, text, work, ("rc", "own"))
        if gc.status != "ok":
            stats.add("gc-" + gc.status)
            if gc.status == "rejected" and stats.counts.get("gc-rejected", 0) <= 30:
                os.makedirs(invalid_dir, exist_ok=True)
                with open(os.path.join(invalid_dir, "%s.ts" % label), "w", newline="\n") as f:
                    f.write(text + "\n/*\n" + short(gc.text, 2000) + "\n*/\n")
            row = [label, gc.summary(), "-", "-"]
        elif gc.run_code != 0:
            stats.add("gc-run-fail")
            os.makedirs(invalid_dir, exist_ok=True)
            with open(os.path.join(invalid_dir, "%s-gc-run.ts" % label), "w", newline="\n") as f:
                f.write(text + "\n/*\n" + short(gc.output, 2000) + "\n*/\n")
            row = [label, gc.summary(), "-", "-"]
        else:
            with open(os.path.join(work, "gc.out"), "w", newline="\n") as f:
                f.write(gc.output)
            stats.add("checked")
            for mm in ("rc", "own"):
                r = res[mm]
                stats.add("%s-%s" % (mm, "rejected" if r.status == "rejected" else "accepted"))
            row = [label, gc.summary(), res["rc"].summary(), res["own"].summary()]
            for mm, kind in sorted(kinds.items()):
                stats.add("FAIL-%s-%s" % (mm, kind))
                sig = (mm, kind, res[mm].run_code if kind == "crash" else None)
                with sig_lock:
                    n = minimized_per_sig.get(sig, 0)
                    minimized_per_sig[sig] = n + 1
                do_min = n < args.minimize_per_signature
                name = "%s-%s-%s" % (label, mm, kind)
                info = process_failure(tc, recheck, text, work, mm, kind, fail_dir, name, do_min)
                failures.append((name, info))
        with tsv_lock:
            tsv.write("\t".join(str(x) for x in row) + "\n")
            tsv.flush()
        if not args.keep_work and not kinds:
            shutil.rmtree(work, ignore_errors=True)
        return gc.status == "ok" and gc.run_code == 0 and res["own"].status == "rejected"

    def one(seed):
        text = Gen(seed, avoid=args.avoid).program()
        if variant(str(seed), text, os.path.join(work_root, str(seed))) and not args.no_repair:
            # own rejected it: delete what own complains about until it accepts, and check that
            work = os.path.join(work_root, "%dr" % seed)
            repaired = own_repair(tc, text, work)
            if repaired:
                stats.add("repaired")
                variant("%dr" % seed, repaired, work)
            else:
                stats.add("repair-failed")
                shutil.rmtree(work, ignore_errors=True)

    seeds = range(args.seed, args.seed + args.count)
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(one, s) for s in seeds]
        for f in concurrent.futures.as_completed(futs):
            f.result()
            done += 1
            if done % 50 == 0 or done == args.count:
                print("[%d/%d %.0fs] %s" % (done, args.count, time.time() - t0,
                                            " ".join("%s=%d" % kv for kv in sorted(stats.counts.items()))),
                      flush=True)
    tsv.close()
    with open(os.path.join(out, "summary.json"), "w", newline="\n") as f:
        json.dump({"tslang": tc.describe(), "seeds": [args.seed, args.seed + args.count - 1],
                   "seconds": round(time.time() - t0), "counts": stats.counts,
                   "failures": sorted(n for n, _ in failures)}, f, indent=2)
    print("failures: %d (see %s)" % (len(failures), fail_dir))
    return 1 if failures else 0


def cmd_replay(args, tc, recheck):
    files = []
    for p in args.files or [os.path.join(HERE, "seeds")]:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".ts"))
        else:
            files.append(p)
    out = os.path.abspath(args.out)
    rows = []
    lock = threading.Lock()

    def one(i, path):
        text = open(path, encoding="utf-8", errors="replace").read()
        work = os.path.join(out, "work", "%04d-%s" % (i, os.path.splitext(os.path.basename(path))[0]))
        gc, res, kinds = check_program(tc, text, work, ("rc", "own"))
        if gc.status != "ok" or gc.run_code != 0:
            row = [path, gc.summary(), "-", "-", "skipped"]
        else:
            row = [path, gc.summary(), res["rc"].summary(), res["own"].summary(),
                   " ".join("%s:%s" % kv for kv in sorted(kinds.items())) or "agree"]
            for mm in kinds:
                with open(os.path.join(work, "%s.out" % mm), "w", newline="\n") as f:
                    f.write(res[mm].output if res[mm].status == "ok" else res[mm].text)
            with open(os.path.join(work, "gc.out"), "w", newline="\n") as f:
                f.write(gc.output)
        with lock:
            rows.append(row)
        if not kinds:
            shutil.rmtree(work, ignore_errors=True)

    os.makedirs(out, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(lambda a: one(*a), enumerate(files)))
    rows.sort()
    with open(os.path.join(out, "replay.tsv"), "w", newline="\n") as f:
        f.write("file\tgc\trc\town\tverdict\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    bad = [r for r in rows if r[4] not in ("agree", "skipped")]
    for r in bad:
        print("\t".join(r))
    print("replayed %d: %d disagree, %d skipped (gc failed)" % (
        len(rows), len(bad), sum(1 for r in rows if r[4] == "skipped")))
    return 1 if bad else 0


def cmd_minimize(args, tc, recheck):
    text = open(args.file, encoding="utf-8").read()
    work = os.path.abspath(os.path.join(args.out, "work", "min-" + os.path.basename(args.file)))
    gc, res, kinds = check_program(tc, text, work, (args.model,))
    if args.model not in kinds:
        print("not failing under %s: gc %s, %s %s" % (args.model, gc.summary(), args.model,
                                                       res[args.model].summary() if args.model in res else "-"))
        return 2
    with open(os.path.join(work, "gc.out"), "w", newline="\n") as f:
        f.write(gc.output)
    name = os.path.splitext(os.path.basename(args.file))[0] + "-" + args.model + "-" + kinds[args.model]
    info = process_failure(tc, recheck, text, work, args.model, kinds[args.model],
                           os.path.join(os.path.abspath(args.out), "failures"), name, True)
    print(json.dumps(info, indent=2))
    print(open(os.path.join(args.out, "failures", name, "repro.ts")).read())
    return 0


def main():
    if IS_WINDOWS:
        # no WER dialog for a crashing test program: 0xC0000374 would otherwise look like a hang
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x8003)
    tslang, lib, gclib = default_paths()
    ap = argparse.ArgumentParser(description="Differential fuzzer for -mm=own and -mm=rc against "
                                             "-mm=gc; the header of this file explains the commands.")
    ap.add_argument("--tslang", default=tslang)
    ap.add_argument("--tslang-lib-path", default=lib)
    ap.add_argument("--gc-lib-path", default=gclib)
    ap.add_argument("--recheck-tslang", action="append", default=[],
                    help="label=path/to/tslang[,lib-path]: also run each minimized repro with this compiler")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "mm-fuzz-out"))
    ap.add_argument("--avoid", type=lambda s: [x for x in s.split(",") if x], default=[],
                    help="comma-separated shapes not to generate, so that a known bug that every "
                         "other program hits stops hiding the rest: " + ", ".join(sorted(AVOIDABLE)))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--count", type=int, default=200)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--minimize-per-signature", type=int, default=8,
                   help="minimize at most this many failures with the same model, kind and exit code")
    r.add_argument("--keep-work", action="store_true")
    r.add_argument("--no-repair", action="store_true",
                   help="do not make an own-accepted variant of a program own rejects")
    p = sub.add_parser("replay")
    p.add_argument("files", nargs="*")
    m = sub.add_parser("minimize")
    m.add_argument("file")
    m.add_argument("--model", choices=("rc", "own"), required=True)
    s = sub.add_parser("show")
    s.add_argument("--seed", type=int, default=1)
    rp = sub.add_parser("report")
    rp.add_argument("dirs", nargs="+", help="output folders of earlier runs")
    args = ap.parse_args()

    if args.cmd == "show":
        sys.stdout.write(Gen(args.seed, avoid=args.avoid).program())
        return 0
    tc = Toolchain(args.tslang, args.tslang_lib_path, args.gc_lib_path)
    recheck = []
    for spec in args.recheck_tslang:
        label, _, rest = spec.partition("=")
        exe, _, libp = rest.partition(",")
        libp = libp or os.path.join(os.path.dirname(os.path.dirname(exe)), "lib")
        recheck.append((label, Toolchain(exe, libp, args.gc_lib_path)))
    return {"run": cmd_run, "replay": cmd_replay, "minimize": cmd_minimize,
            "report": cmd_report}[args.cmd](args, tc, recheck)


if __name__ == "__main__":
    sys.exit(main())
