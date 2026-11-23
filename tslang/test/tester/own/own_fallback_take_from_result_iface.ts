// -mm=own --own-fallback=count: a field of a type that is not one pointer (an interface: `this` and
// its vtable) taken out of a call's own result that another owner also holds. `mk` keeps the box in a
// global and returns it, which own refuses ('b' is used after its value was moved); with the flag the
// box is counted. The take (ts.TakeOutOfField) leaves the field to the global and gives what it holds
// an owner through the interface's counted retain, so the global still reads it.
interface Shape { v: number; }
class Node { constructor(public v: number) {} }
class Box { shape: Shape | null = null; }

let last: Box | null = null;

function mk(v: number) {
    const b = new Box();
    b.shape = new Node(v);
    last = b;
    return b;
}

function takeTemp(v: number) {
    return mk(v).shape;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const s = takeTemp(i);
        sum += s!.v;
        sum += last!.shape ? last!.shape!.v : -1000000;
    }
    print(sum);
    print("done.");
}
