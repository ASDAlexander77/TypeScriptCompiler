// A local declared in a generator's loop is a field of the state object, and its declaration runs
// again on every turn, into the same field. What the last turn stored there is given back first;
// before, nothing gave it back, and 500000 runs of `untilBreak` grew by 150 MB under rc and own.
// The leak itself shows only in measure.ps1 (and generator-loop-local-release.cmake reads the IR);
// this checks that the release frees the old value, not the new one: each local is read after a
// churn(), which reuses freed blocks, so a value freed too early reads garbage.
//
// Under own the release makes `next()` a call that may drop, and a caller holds what it borrowed
// out of one record across the next call: a for...of over a generator, and two records by hand.
//
// Strings are made in another function: a temporary made in a generator's own body is never
// released (spec 18.5), which would hide the leak this covers.
function label(i: number) {
    return "s" + i;
}

function churn() {
    const junk: string[][] = [];
    for (let i = 0; i < 32; i++) {
        junk.push([label(i), label(i + 1)]);
    }
}

// a `const` in a `for`, read after a churn() on its own turn
function* rows(n: number) {
    for (let i = 0; i < n; i++) {
        const row: string[] = [label(i), label(i + 1)];
        churn();
        yield row.length + row[0].length;
    }
}

// a `const` in a `while (true)`, the last one left by `break`: the state's release gives it back
function* untilBreak() {
    let i = 0;
    while (true) {
        const result: string[] = [label(i++), "x"];
        if (i > 3) {
            break;
        }

        yield result.length;
    }
}

// yields a fresh array each turn while a local is given back in the same `next()`
function* pairs() {
    let i = 0;
    while (i < 3) {
        const keep: string[] = [label(i), "k"];
        i++;
        churn();
        yield [keep.length + keep[0].length + i, 7];
    }
}

function main() {
    let t = 0;
    for (let n = 0; n < 2000; n++) {
        for (const v of rows(3)) {
            t += v;
        }

        for (const v of untilBreak()) {
            churn();
            t += v;
        }

        // the loop variable borrows the record's value across the next `next()`
        for (const g of pairs()) {
            churn();
            t += g[0] + g.length;
        }

        // by hand: the first record is read after the second `next()`
        const ps = pairs();
        const a = ps.next();
        const b = ps.next();
        churn();
        t += a.value[0] + b.value[0];

        // given up after the first value: the state still gives back its local
        const half = rows(3);
        t += half.next().value;
    }

    // rows 4 + 4 + 4, untilBreak 2 + 2 + 2, pairs (5 + 2) + (6 + 2) + (7 + 2), by hand 5 + 6, half 4
    assert(t == 2000 * (12 + 6 + 24 + 11 + 4), "t");
    print("done.");
}
