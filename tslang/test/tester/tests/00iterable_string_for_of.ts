// for...of over an iterator interface of strings (`cat(v: Iterable<string>)`), given an array, a string or a
// generator. Under -mm=own it was "a value takes a second reference": next() called through the interface was
// not known to hand over what it returns. Every function behind it is a generator's own next, which moves what
// it yields into the record it returns, so the loop owns each string (TSLANG 157). The array literal given
// straight to the call (`cat(["x" + i])`) leaked under -mm=own, with its strings: nothing gave it back.
type IterateResult<T> = { value: T, done: boolean };
interface Iterable<T> {
    next: () => IterateResult<T>;
}

interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
}

function cat(v: Iterable<string>) {
    let s = "";
    for (const c of v) {
        s += c;
    }

    return s;
}

// the strings kept: each is the loop's to give
function keep(v: Iterable<string>) {
    const out: string[] = [];
    for (const c of v) {
        out.push(c);
    }

    return out;
}

// left early: what is not iterated goes with the iterator
function first(v: Iterable<string>) {
    for (const c of v) {
        if (c.length > 1) {
            return c;
        }
    }

    return "none" + 0;
}

function count(v: Iterable<string>) {
    let n = 0;
    for (const c of v) {
        if (c == "y2") {
            break;
        }

        n++;
    }

    return n;
}

// by hand, as the loop does
function second(v: Iterable<string>) {
    let r = v.next();
    r = v.next();
    return r.done ? "none" + 0 : r.value;
}

function* words(n: number): Generator<string> {
    for (let i = 0; i < n; i++) {
        yield "w" + i;
    }
}

function main() {
    for (let round = 0; round < 1000; round++) {
        assert(cat(["x" + round, "y" + 2]) == "x" + round + "y2");
        assert(cat("abc" + round) == "abc" + round);
        assert(cat(words(3)) == "w0w1w2");

        const items = ["x" + 1, "y" + 2, "z" + round];
        const k = keep(items);
        assert(k.length == 3 && k[0] == "x1" && k[2] == "z" + round);
        // the array iterated is still the caller's
        assert(items.length == 3 && items[0] == "x1" && items[2] == "z" + round);
        const k2 = keep(words(2));
        assert(k2.length == 2 && k2[1] == "w1");
        const k3 = keep("h" + round);
        assert(k3[0] == "h");

        assert(first(items) == "x1" && first("abc") == "none0" && first(words(2)) == "w0");
        assert(count(items) == 1 && count(words(5)) == 5);
        assert(second(items) == "y2" && second(words(1)) == "none0" && second("ab") == "b");
    }

    print("done.");
}
