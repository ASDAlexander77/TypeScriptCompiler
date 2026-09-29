// -mm=own across a module boundary: calls into a library that exported `__own_no_drops` keep a
// borrow alive. `v` borrows a field of the parameter `l`, and a call that may destroy anything
// reachable from `l` ends it; `total` and `M.first` are known, from the library, to destroy
// nothing. Built against a library that does not say so, `v.length` is an error.
import './export_own_no_drops'

class L {
    v: number[] = [1, 2, 3];
}

function read(l: L) {
    const v = l.v;
    const t = total(v) + M.first(v);
    return v.length + t;
}

function main() {
    const l = new L();
    assert(read(l) == 10);
    print("done.");
}
