// A keeper's push of a value this function did not make and is no parameter of it - a field's
// function value - cannot be made immortal: only its maker knows what it relies on.
type Callback = (n: number) => void;
let retained: Callback[] = [];

class H {
    cb: Callback = (n: number) => {};
}

function keep(h: H) {
    retained.push(h.cb);
}

function main() {
    keep(new H());
}
