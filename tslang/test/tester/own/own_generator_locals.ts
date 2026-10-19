// -mm=own: what a generator does with its locals, which are fields of its state object.
// - A local narrowed by a test is a view of the local, read again at each use - after a yield too -
//   not a second field holding a second reference.
// - A spread (`[...a]`) of a local builds its array in the frame; the array moves into its field.
// - A local read only to be yielded moves its value to the caller: the yield takes it out, and
//   neither the state's release nor the next value stored there gives it back.
// churn() between yields reuses freed blocks, so a value freed too early reads garbage.
class Source {
    n = 0;

    next(): string[] | null {
        if (this.n >= 3) {
            return null;
        }

        const r: string[] = ["a" + this.n, "b" + this.n];
        this.n++;
        return r;
    }
}

function churn() {
    const junk: string[][] = [];
    for (let i = 0; i < 32; i++) {
        junk.push(["j" + i, "k" + i]);
    }
}

// narrowed before a yield, read after it
function* lengths(src: Source) {
    while (true) {
        const r = src.next();
        if (r == null) {
            break;
        }

        yield 1;
        churn();
        yield r.length + r[0].length;
    }
}

// a spread of a narrowed local, yielded
function* copies(src: Source) {
    while (true) {
        const r = src.next();
        if (r == null) {
            break;
        }

        const groups = [...r];
        churn();
        yield groups;
    }
}

function main() {
    let t = 0;
    for (let i = 0; i < 2000; i++) {
        for (const v of lengths(new Source())) {
            churn();
            t += v;
        }

        for (const g of copies(new Source())) {
            churn();
            t += g.length + g[0].length + g[1].length;
        }

        // given up after one value: the state still releases what it owns
        const half = copies(new Source());
        t += half.next().value.length;
    }

    assert(t == 2000 * (15 + 18 + 2), "t");
    print("done.");
}
