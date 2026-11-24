// Under -mm=own a function value read at run time is given as another function type through an adapter that holds
// it in a box (TSLANG 188). A parameter's value adapted for a local works; adapted and returned, the box would keep
// the caller's value past the call: the error says so.
type H = (() => void) | (() => number);
function reg(cb: () => number): H {
    let u: H = cb;
    return u;
}
