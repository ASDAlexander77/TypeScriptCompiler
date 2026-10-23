// A Shared<T> handle read for an argument, and given a new value by a later argument before the call
// takes it: the assignment gave back the count the old handle held, the last one, and the callee
// read the freed block. The caller holds a count across the call, in every model that counts. Each
// read follows allocations of the same size, so a freed block is reused.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
}

class Junk {
    a = -1;
    b = -2;
    tag = "junk";
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Junk());
    return keep.length;
}

function g(s: Shared<V>, b: number) {
    churn();
    assert(s.value.tag == "v" + s.value.x, "the handle's block");
    return s.value.x * 10 + b;
}

function main() {
    let s = new Shared(new V(1));
    assert(g(s, (s = new Shared(new V(2)), 0)) == 10, "a handle given a new value in the next argument");
    churn();
    assert(s.value.x == 2 && s.value.tag == "v2", "the handle's new value");
    print("done.");
}
