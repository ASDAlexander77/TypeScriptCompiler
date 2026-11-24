// TSLANG 181: a field read through a union of object shapes, returned, borrows the union (164); a getter of it is
// read and called as the getter of one shape is
class Box {
    n = 1;
    t = "b" + 1;
}

type Sq = { kind: "sq"; size: number; box: Box };
type Ci = { r: number; kind: "ci"; box: Box };

function getBox(s: Sq | Ci): Box {
    return s.box;
}

function sumOf(i: number) {
    if (i % 2 == 0) {
        const s: Sq | Ci = { kind: "sq", size: 2, box: new Box() };
        return getBox(s).n + getBox(s).t.length;
    }

    const c: Sq | Ci = { r: 1.5, kind: "ci", box: new Box() };
    const b = getBox(c);
    return b.n * 10;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 1000; i++) t += sumOf(i);
    assert(t == 6500);
    print("done.");
}
