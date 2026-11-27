// TSLANG 217: the -mm=own refusals of 56, 128, 159, Constructor<T> of `...args: any[]` and 216's "not yet", which
// compile and run under every memory model now; also a rest `any[]` given nothing, and `new c(a, b)` through an
// interface's construct signature with a rest parameter (every model).

// The tests compile without the default library: its ObjectDictionary, as the compiler finds it by name (as in
// 00tslang_issue_216.ts), and the Generator interface the generators are given
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
}

@dictionary
class ObjectDictionary<T = any> {
    private _keys: string[] = [];
    private _values: T[] = [];

    get size() {
        return this._keys.length;
    }

    private find(key: string): int {
        for (let i = 0; i < this._keys.length; i++) {
            if (this._keys[i] == key) {
                return i;
            }
        }

        return -1;
    }

    get(key: string): T | undefined {
        const i = this.find(key);
        if (i < 0) {
            return undefined;
        }

        return this._values[i];
    }

    set(key: string, value: T | undefined) {
        if (value === undefined) {
            this.delete(key);
            return;
        }

        const i = this.find(key);
        if (i < 0) {
            this._keys.push("" + key);
            this._values.push(value);
            return;
        }

        this._values[i] = value;
    }

    [key: string]: T | undefined;

    has(key: string): boolean {
        return this.find(key) >= 0;
    }

    delete(key: string): boolean {
        const i = this.find(key);
        if (i < 0) {
            return false;
        }

        this._keys.splice(i, 1);
        this._values.splice(i, 1);
        return true;
    }

    keys(): string[] {
        const r: string[] = [];
        for (let i = 0; i < this._keys.length; i++) {
            r.push("" + this._keys[i]);
        }

        return r;
    }
}


// 56: a `let` walking fields - `cur = this.b` before `this.a` is given another object, in every round of the loop:
// what `cur` reads after it is `this.b`
class V {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

class Walker {
    a: V = new V(1);
    b: V = new V(2);
    run(): number {
        let cur = this.a;
        let s: number = 0;
        for (let i = 0; i < 2; i++) {
            s += cur.x;
            cur = this.b;
            this.a = new V(5);
        }

        return s + cur.x;
    }
}

// 128: a `for (let i ...)` whose body assigns `i` before its closures, and a string loop variable
function closuresOfAssignedLoopVariable(): number {
    const fs: (() => number)[] = [];
    for (let i = 0; i < 6; i++) {
        i++;
        fs.push(() => i);
    }

    let s = 0;
    for (const f of fs) s = s * 10 + f();
    return s;
}

function closuresOfStringLoopVariable(k: number): string {
    const fs: (() => string)[] = [];
    for (let s = "a" + k; s.length < 5; s = s + "b") {
        fs.push(() => s);
    }

    let all = "";
    for (const f of fs) all += f() + ",";
    return all;
}

// 159: `yield* this.range()` in a generator method that borrows `this`, and `yield*` of a generator over a parameter
class Range {
    n: number = 3;

    @borrows("this")
    *range(): Generator<number> {
        for (let i = 0; i < this.n; i++) yield i;
    }

    @borrows("this")
    *all(): Generator<number> {
        yield 100;
        yield* this.range();
        yield 200;
    }
}

// ...and one that says nothing: a generator method of a class this module keeps to itself borrows its `this` (user
// ruling 2026-11-27), so the object is still its caller's after `r.range()`
class Plainrange {
    n: number = 3;
    *range(): Generator<number> {
        for (let i = 0; i < this.n; i++) yield i;
    }

    *all(): Generator<number> {
        yield 100;
        yield* this.range();
    }
}

function* up(xs: number[]): Generator<number> {
    for (const x of xs) yield x;
}

function* around(xs: number[]): Generator<number> {
    yield 100;
    yield* up(xs);
    yield 200;
}

// Constructor<T> of `new (...args: any[]) => T`: a class given to the interface, called with nothing and with arguments
interface Constructor<T> {
    new (...args: any[]): T;
}

interface MakesWith {
    new (a: number, ...rest: any[]): { n: number };
}

class Plain {
    a = 1;
}

class Two {
    n: number;
    s: string;
    constructor(n: number, s: string) {
        this.n = n;
        this.s = s;
    }
}

class Sum {
    n: number;
    constructor(a: number, b: string) {
        this.n = a + b.length;
    }
}

// 216: a function that makes the dictionary it fills says to its callers that it drops nothing of theirs
function count(items: string[]): Record<string, number> {
    const out: Record<string, number> = {};
    for (const it of items) out[it] = 1;
    return out;
}

function countEach(items: string[]): number {
    let n = 0;
    for (const it of items) {
        const d = count(items);
        n += d[it];
    }

    return n;
}

// a rest `any[]` given no argument
function restCount(...a: any[]): number {
    return a.length + 2;
}

function main() {
    assert(new Walker().run() == 5);

    assert(closuresOfAssignedLoopVariable() == 135);
    assert(closuresOfStringLoopVariable(7) == "a7,a7b,a7bb,");

    const r = new Range();
    let s: number = 0;
    for (const v of r.all()) s += v;
    assert(s == 303);
    const it = r.all();
    it.next();
    it.next();
    it.return(0);

    const pr = new Plainrange();
    s = 0;
    for (const v of pr.all()) s += v;
    for (const v of pr.range()) s += v;
    assert(s == 106 && pr.n == 3);

    s = 0;
    for (const v of around([1, 2, 3])) s += v;
    assert(s == 306);

    const c: Constructor<Plain> = Plain;
    const c2: Constructor<Two> = Two;
    const m: MakesWith = Sum;
    let total = 0;
    for (let i: number = 0; i < 100; i++) {
        total += new c().a;
        const t = new c2(i, "x" + i);
        total += t.n + t.s.length;
        total += new m(i, "ab" + i).n;
    }
    assert(total == 100 + 4950 + 290 + 4950 + 390);

    const xs: string[] = [];
    for (let i = 0; i < 4; i++) xs.push("k" + i);
    assert(countEach(xs) == 4);

    let rest = 0;
    for (let i = 0; i < 10; i++) rest += restCount();
    assert(rest == 20);

    print("done.");
}
