// A forward reference in another module's declarations: `A` is printed before the `B` it holds.
import './export_forward_iface'

function main() {
    let sum: number = 0;
    for (let i = 0; i < 100; i++) {
        const a = mkA(i);
        sum += a.b.k + a.n;
    }

    assert(sum == 54450, "a.b.k and a.n of an imported A");

    const a = mkA(3);
    assert(a.b.k == 30 && a.n == 3, "the fields of one A");
    print("done.");
}
