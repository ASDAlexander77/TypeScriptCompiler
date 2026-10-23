// -mm=own: `Maker` (export_own_record.ts) is @final, so `m.rebase = ...` is a call of the setter the
// library defines, and the library does not list it as destroying nothing: it replaces the node the
// record borrows. A read of the record after it is refused.
import './export_own_record'

function main() {
    const a = new Node(1);
    const m = new Maker();
    const q = m.pairWith(a);
    m.rebase = new Node(9);
    print(q.a.v);
}
