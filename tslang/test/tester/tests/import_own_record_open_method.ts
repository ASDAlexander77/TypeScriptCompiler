// -mm=own: `Open` (export_own_record.ts) is not @final, so a call of `look` may run an override the
// importer cannot see, one that destroys what the record borrows: although the library lists
// `Open.look` as destroying nothing, a read of the record after the call is refused.
import './export_own_record'

function main() {
    const a = new Node(1);
    const b = new Node(2);
    const o = new Open();
    const p = pairOf(a, b);
    print(o.look(a));
    print(p.a.v);
}
