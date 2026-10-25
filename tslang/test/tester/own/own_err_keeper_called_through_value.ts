// A function that keeps a parameter for good, passed on as a function value: whoever calls it
// through that value would not know to make the argument immortal.
type Callback = (n: number) => void;
let retained: Callback[] = [];

function keep(h: Callback) {
    retained.push(h);
}

function apply(f: (h: Callback) => void) {
    f((n: number) => {});
}

function main() {
    apply(keep);
}
