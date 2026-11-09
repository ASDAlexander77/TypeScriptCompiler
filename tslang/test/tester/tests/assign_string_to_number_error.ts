// A string given to a number (a field here) is an error, as in TypeScript (TS2322). It was
// converted: "x" became 0, and the mistake was silent.
class Counter {
    count: number = 0;
}

function main() {
    const c = new Counter();
    c.count = "x";
    print(c.count);
}
