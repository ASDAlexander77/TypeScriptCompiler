// -mm=own, without --own-fallback=count: a field taken out of a call's own result. The take is a
// ts.TakeOutOfField in every own build (own-take-out-shape.cmake): a callee built with the flag in
// another module may return an object with extra owners, and the take then leaves the field to them.
// Here nothing is counted: the field is emptied at run time, and the program prints what gc prints.
class Node { constructor(public v: number) {} }
class Box { node: Node | null = null; }

function mk(v: number) {
    const b = new Box();
    b.node = new Node(v);
    return b;
}

function takeTemp(v: number) {
    return mk(v).node;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        const n = takeTemp(i);
        s += n!.v;
    }

    print(s);
    print("done.");
}
