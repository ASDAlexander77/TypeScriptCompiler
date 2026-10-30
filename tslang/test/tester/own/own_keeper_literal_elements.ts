// -mm=own: a keeper - an append-only array global - makes what is pushed onto it immortal, so a read
// of its elements owns nothing and any release skips it. The elements its own literal is made with
// were not made immortal: `const GA: N[] = [mk(5)]` was taken for a keeper all the same, and a local
// given `GA[0]` freed it when the local's block ended (mm_fuzz seeds 70040-70386: 9 of 10 own
// failures of one round). An array literal of anything but constants makes no keeper now, and its
// elements are borrowed like any array's: storing one in an array or a global is an error under own.
// Each value is read after churn() has reused anything freed too early.
class N {
    items: number[] = [5];
    constructor(public x: number) {}
}

function mk(i: number): N {
    return new N(i);
}

function churn() {
    const hold: N[] = [];
    for (let i = 0; i < 64; i++) {
        hold.push(new N(777));
    }
}

interface IX {
    x: number;
}

const GA: N[] = [mk(5)];

function main() {
    for (let i = 0; i < 3; i++) {
        let n = GA[0];
    }

    churn();
    assert(GA[0].x == 5 && GA[0].items[0] == 5, "a local in a loop leaves the element");

    {
        let n = GA[0];
        let ix: IX = GA[0];
    }

    churn();
    assert(GA[0].x == 5, "a local and an interface in a block leave the element");

    GA.push(mk(6));
    churn();
    assert(GA[1].x == 6, "a pushed element");

    print("done.");
}
