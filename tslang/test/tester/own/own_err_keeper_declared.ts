// `@immortalizes` is checked against the body: it names 'a', but the body keeps 'b' for good.
type Callback = (n: number) => void;
let retained: Callback[] = [];

@immortalizes("a")
function keep(a: Callback, b: Callback) {
    retained.push(b);
}

function main() {
    keep((n: number) => {}, (n: number) => {});
}
