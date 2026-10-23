// -mm=own: the record `pairOf` returns (export_own_record.ts) borrows `a`, so a read of it after `a`
// is given a new value is refused.
import './export_own_record'

function main() {
    let a = new Node(1);
    const b = new Node(2);
    const p = pairOf(a, b);
    a = new Node(3);
    print(p.a.v);
}
