// A borrowing record from another module (export_own_record.ts): the importer owns each box
// `pairOf` and `pairWith` return, and releases it at the end of each iteration, leaving the borrowed
// nodes to their owners. Each record is read after a call to `Maker.weigh`, and the second after
// `Maker.weight` is set and read too, which the library lists as destroying nothing: `Maker` is
// @final, so a call through its vtable is to that method or accessor and no other. Under gc and rc
// the markers mean nothing.
import './export_own_record'

function main() {
    const a = new Node(1);
    const b = new Node(2);
    const m = new Maker();
    let sum = 0;
    for (let i = 0; i < 1000; i++) {
        const p = pairOf(a, b);
        sum += m.weigh(a);
        sum += p.a.v + p.b.v;
    }

    for (let i = 0; i < 1000; i++) {
        const q = m.pairWith(a);
        sum += m.weigh(b);
        m.weight = i;
        sum += m.weight;
        sum += q.a.v + q.b.v;
    }

    assert(sum == 27000, "records across the boundary");
    assert(a.v == 1 && b.v == 2 && m.base.v == 5 && m.w == 999, "the borrowed nodes outlive the records");
    print("done.");
}
