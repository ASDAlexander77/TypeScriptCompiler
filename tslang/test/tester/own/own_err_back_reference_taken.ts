// A field a back-reference is stored into (spec 70) is never taken out of its object: its closure
// borrows the object, which the function releases when it returns.
class Counter {
    n: number = 1;
    get = () => this.n;
}

function take() {
    const c = new Counter();
    return c.get;
}

function main() {
    print(take()());
}
