// -mm=own across a module boundary rejects: `shrink` removes an element of the array it is
// given, and the library does not list it as destroying nothing, so `w` may be gone.
import './export_own_no_drops'

class L {
    all: number[][] = [[4], [5]];
}

function read(l: L) {
    const w = l.all[1];
    shrink(l.all);
    return w.length;
}

function main() {
    print(read(new L()));
}
