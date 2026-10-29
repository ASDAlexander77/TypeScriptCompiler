// A back-reference read out of its object is a borrow of the object (spec 70): it cannot be stored
// into another object, which may outlive it.
class Counter {
    n: number = 1;
    get = () => this.n;
}

class Holder {
    call: () => number = () => 0;
}

function main() {
    const c = new Counter();
    const h = new Holder();
    h.call = c.get;
    print(h.call());
}
