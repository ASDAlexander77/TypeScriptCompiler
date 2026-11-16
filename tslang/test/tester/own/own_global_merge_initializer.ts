// A global initialized by `?:` of a fresh object and null or undefined: the global owns what the way
// taken makes; the other way gives null, which owns nothing.
class Node {
    constructor(public v: number) {}
}

function on() {
    return true;
}

function off() {
    return false;
}

function mk(n: number) {
    return new Node(n);
}

let pick: Node | null = on() ? mk(3) : null;
let none: Node | null = off() ? mk(4) : null;
let maybe: Node | undefined = on() ? mk(5) : undefined;

function main() {
    assert(pick !== null && pick.v == 3, "pick");
    assert(none === null, "none");
    assert(maybe !== undefined && maybe.v == 5, "maybe");
    pick = mk(6);
    assert(pick.v == 6, "reassigned");
    print("done.");
}
