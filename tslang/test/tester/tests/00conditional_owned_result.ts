// regression test: a conditional expression whose branches disagree about ownership.
//
// `cond ? borrowed : f()` builds a `ts.If` with a result, and its two branches hand back
// references on different terms - one borrowed from wherever it already lives, one a fresh +1
// that somebody has to take over. The receiver outside can only retain, so the +1 from the
// allocating branch was left with no owner and leaked every time that branch was taken.
// §9.30 could not reach it: the value's only user is `ts.Result`, a terminator, and releasing
// a value handed to a successor would free it while it is still live.
//
// It is not a corner of the language. In `raytrace.ts` a single line of this shape -
//     let reflectedColor = (depth >= this.maxDepth) ? Color.grey : this.getReflectionColor(...)
// - held 18 of the 20 MB reference counting had left on the table, and rewriting just that line
// as `if`/`else` reclaimed it. See docs/reference-counting-evaluation.md section 9.73.
//
// The fix makes the branches agree before anything outside looks: the borrowing branch takes a
// reference of its own, so every branch yields +1 and the `ts.If` result carries a reference
// the way a call's result does. That direction matters - releasing the owned branch instead
// would be an over-release, because at no point inside the region has the receiver retained
// yet. So this test holds on to results from BOTH branches, allocates hard over anything
// wrongly freed, and only then reads them back: a freed block keeps its contents until
// something reuses it, which is what makes the churn load-bearing.

class Box {
    tag: string;
    n: number;
    constructor(tag: string, n: number) {
        this.tag = tag;
        this.n = n;
    }
}

const shared = new Box("shared", -1.0);
const other = new Box("other", -2.0);

function make(n: number): Box {
    return new Box("fresh", n);
}

// the raytrace shape: borrowed on the left, freshly allocated on the right
function pick(useShared: boolean, n: number): Box {
    return useShared ? shared : make(n);
}

// the same with the branches the other way round
function pickReversed(useFresh: boolean, n: number): Box {
    return useFresh ? make(n) : shared;
}

// both branches allocate
function pickBoth(first: boolean, n: number): Box {
    return first ? make(n) : make(n + 1.0);
}

// neither branch allocates - must keep working exactly as before
function pickNeither(first: boolean): Box {
    return first ? shared : other;
}

// nested, so the inner conditional's result is itself a branch of the outer one
function pickNested(a: boolean, b: boolean, n: number): Box {
    return a ? shared : (b ? make(n) : other);
}

// a string rather than an object, since strings are their own owning type
function pickString(useLiteral: boolean, n: number): string {
    return useLiteral ? "literal" : `fresh ${n}`;
}

function main() {
    // Hold results of every shape, taking both branches of each.
    let kept: Box[] = [];
    let keptStrings: string[] = [];
    for (let i = 0; i < 12; i++) {
        const even = i % 2 == 0;
        kept.push(pick(even, i));
        kept.push(pickReversed(even, i));
        kept.push(pickBoth(even, i));
        kept.push(pickNeither(even));
        kept.push(pickNested(i % 3 == 0, even, i));
        keptStrings.push(pickString(even, i));
    }

    // Reuse anything that was released too early.
    let churn = 0.0;
    for (let i = 0; i < 20000; i++) {
        churn = churn + pick(i % 2 == 0, i).n + pickBoth(i % 2 == 0, i).n;
        churn = churn + pickString(i % 2 == 0, i).length;
    }

    // A conditional whose result nothing receives at all: the discarded-temporary path.
    for (let i = 0; i < 20000; i++) {
        make(i);
        pick(i % 2 == 0, i);
    }

    let bad = 0;
    for (let i = 0; i < 12; i++) {
        const even = i % 2 == 0;
        const at = i * 5;

        // pick: shared on even, fresh on odd
        if (even) {
            if (kept[at].tag != "shared" || kept[at].n != -1.0) bad = bad + 1;
        } else {
            if (kept[at].tag != "fresh" || kept[at].n != i) bad = bad + 1;
        }

        // pickReversed: fresh on even, shared on odd
        if (even) {
            if (kept[at + 1].tag != "fresh" || kept[at + 1].n != i) bad = bad + 1;
        } else {
            if (kept[at + 1].tag != "shared" || kept[at + 1].n != -1.0) bad = bad + 1;
        }

        // pickBoth: always fresh, n differs by branch
        if (kept[at + 2].tag != "fresh") bad = bad + 1;
        if (kept[at + 2].n != (even ? i : i + 1.0)) bad = bad + 1;

        // pickNeither: two different borrowed globals
        if (kept[at + 3].tag != (even ? "shared" : "other")) bad = bad + 1;

        // pickNested
        if (i % 3 == 0) {
            if (kept[at + 4].tag != "shared") bad = bad + 1;
        } else if (even) {
            if (kept[at + 4].tag != "fresh" || kept[at + 4].n != i) bad = bad + 1;
        } else {
            if (kept[at + 4].tag != "other") bad = bad + 1;
        }

        if (keptStrings[i] != (even ? "literal" : `fresh ${i}`)) bad = bad + 1;
    }

    assert(bad == 0, "a conditional's result was freed while still referenced");

    // the globals must still be intact after all of that
    assert(shared.tag == "shared", "shared global survived");
    assert(other.tag == "other", "other global survived");
    assert(churn > 0.0, "churn");

    print("done.");
}
