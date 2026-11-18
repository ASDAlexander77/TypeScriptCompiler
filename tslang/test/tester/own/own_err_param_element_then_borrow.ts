// A parameter put into an array literal and then given one of its elements is kept: a caller that
// reads its argument after the call reads a value it gave up (fuzz 958275)
class V {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

function intoArray(p: V): number {
    const a: V[] = [new V(1), p];
    p = a[0];
    return p.x;
}

function main() {
    const c = new V(3);
    print(intoArray(c));
    print(c.x);
}
