// Under -mm=own a function value read at run time is not given as another function type yet (an adapter would hold
// a second reference to it in a box): the error says so.
type H = (() => void) | (() => number);
function reg(cb: () => number) {
    let u: H = cb;
    u();
}
