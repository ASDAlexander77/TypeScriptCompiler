// Aliases that mention another alias - or the same one, one after another rather than inside
// itself - are not circular (issue #370 made a type alias meeting itself an error).
type P = [x: number, y: number];
type Seg = [from: P, to: P];
type Pair<T> = [first: T, second: T];
type Box<T> = { v: T };
type Nested = Pair<Pair<number>>;
type Boxes<T> = Box<Box<T>>;

interface Node {
    value: number;
    next?: Node;
}

function main() {
    const s: Seg = [[1, 2], [3, 4]];
    const n: Nested = [[1, 2], [3, 4]];
    const b: Boxes<number> = { v: { v: 7 } };
    const tail: Node = { value: 2 };
    const head: Node = { value: 1, next: tail };
    assert(s[1][0] == 3);
    assert(n[1][1] == 4);
    assert(b.v.v == 7);
    assert(head.next.value == 2);
    print("done.");
}
