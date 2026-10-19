// -mm=own: `@lends("p")` says the result is a borrow of the argument `p` - the caller owns nothing
// of it, and it may not outlive `p`. An exported function says it in its signature, since a caller
// in another module cannot see the body; without the marker it is rejected ("it can be called from
// another module"). Under gc and rc the marker means nothing.
export class Node {
    constructor(public v: number) {}
}

@final
export class List {
    first: Node = new Node(1);
    second: Node = new Node(2);

    @lends("this")
    head(): Node {
        return this.first;
    }
}

@lends("l")
export function firstOf(l: List): Node {
    return l.first;
}

// a parameter other than the first
@lends("l")
export function secondOf(n: number, l: List): Node {
    return n > 0 ? l.second : l.second;
}

// keeps one argument and lends another
@keeps("n")
@lends("l")
export function swapFirst(n: Node, l: List): Node {
    l.second = n;
    return l.first;
}

class Vec {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Vec(999);
    }
}

function main() {
    const l = new List();
    const h = l.head();
    const f = firstOf(l);
    const s = secondOf(1, l);
    churn();
    assert(h.v == 1 && f.v == 1 && s.v == 2);

    const kept = swapFirst(new Node(3), l);
    churn();
    assert(kept.v == 1 && l.second.v == 3 && l.first.v == 1);

    print("done.");
}
