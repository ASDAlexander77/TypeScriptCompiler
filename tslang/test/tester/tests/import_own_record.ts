// A borrowing record from another module (export_own_record.ts): the importer owns each box
// `pairOf` and `pairWith` return, and releases it at the end of each iteration, leaving the borrowed
// nodes to their owners. Each loop calls one of them: a call to another module's method may destroy
// what it can reach, so a record cannot be read across one. Under gc and rc the markers mean nothing.
import './export_own_record'

function main() {
    const a = new Node(1);
    const b = new Node(2);
    let sum = 0;
    for (let i = 0; i < 1000; i++) {
        const p = pairOf(a, b);
        sum += p.a.v + p.b.v;
    }

    const m = new Maker();
    for (let i = 0; i < 1000; i++) {
        const q = m.pairWith(a);
        sum += q.a.v + q.b.v;
    }

    assert(sum == 9000, "records across the boundary");
    assert(a.v == 1 && b.v == 2 && m.base.v == 5, "the borrowed nodes outlive the records");
    print("done.");
}
