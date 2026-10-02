// -mm=own, phase 7: a generator's state object is an ordinary owned block. Its maker makes it and
// returns it, the caller owns it, and its release destroys whatever the generator's locals hold.
// Each case runs many times, so a state object that is never freed shows in measure.ps1.

function* evens() {
    yield 2;
    yield 4;
}

// an object literal with a method: the same made block, seen as its object
function makeCounter(start: number) {
    return {
        n: start,
        next() {
            this.n++;
            return this.n;
        },
    };
}

function main() {
    let t = 0;
    for (let i = 0; i < 20000; i++) {
        // through for...of
        for (const v of evens()) {
            t += v;
        }

        // by hand, the generator kept in a local
        const it = evens();
        let r = it.next();
        while (!r.done) {
            t += r.value;
            r = it.next();
        }

        const c = makeCounter(i % 10);
        t += c.next() + c.next();
    }

    assert(t == 480000, "t");
    print("done.");
}
