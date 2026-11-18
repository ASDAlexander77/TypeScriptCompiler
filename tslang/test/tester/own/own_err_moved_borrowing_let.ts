// A parameter given a borrowed element on one path and moved somewhere that keeps it on another:
// -mm=own cannot tell whether the parameter owns what it holds (fuzz 949146, which freed the element).
class V {
    x: number;
    items: V[] = [];
    constructor(x: number) {
        this.x = x;
    }
}

class Sink {
    items: V[] = [];
}

const sink = new Sink();

function f2(p3: V, p4: V): number {
    if (p4.x > 7) {
        p4 = p3.items[0];
    } else {
        sink.items.push(p4);
    }

    return p3.x;
}

function main() {
    const a = new V(9);
    a.items.push(new V(1));
    print(f2(a, new V(3)));
}
