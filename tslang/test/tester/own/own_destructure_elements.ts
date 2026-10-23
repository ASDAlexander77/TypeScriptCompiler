// `let [a, b] = pair` reads each element behind a test of the array's length whose other side is
// the default. Under own that merge was refused ("merged from several branches"), though the plain
// read `let a = pair[0]` is a borrow of the element: the destructured local is that borrow, and a
// string is copied, as a string read out of an element always is (spec 22). Each read follows
// allocations of the same size, so a freed block would be reused.
class N {
    tag = "";
    constructor(public x: number) {
        this.tag = "n" + x;
    }
}

function churn() {
    const keep: N[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new N(777));
    return keep.length;
}

function mk(i: number) {
    return "elem" + i;
}

function classes() {
    let pair = [new N(1), new N(2)];
    let [a, b] = pair;
    churn();
    assert(a.tag == "n1" && b.tag == "n2", "class elements");
    return a.x + b.x;
}

function strings() {
    let s = [mk(1), mk(2)];
    let [x, y] = s;
    s = [mk(9)];
    churn();
    assert(x == "elem1" && y == "elem2", "strings outlive their array");
    return x.length + y.length;
}

function inLoop() {
    let pair = [new N(3), new N(4)];
    let sum = 0;
    for (let i = 0; i < 3; i++) {
        let [a, b] = pair;
        churn();
        sum += a.x * b.x;
    }

    return sum;
}

function main() {
    assert(classes() == 3, "classes");
    assert(strings() == 10, "strings");
    assert(inLoop() == 36, "in a loop");
    print("done.");
}
