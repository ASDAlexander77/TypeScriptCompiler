// -mm=own across a module boundary: an importer builds objects of the library's classes. `new H()`
// calls the constructor through a method bound to the new object - `ts.CreateBoundFunction` over
// the object cast to `!ts.opaque` - and that was read as taking the object, so every later use of
// it was "used after its value was moved". It reads the object the way calling a method of this
// module does. Each object is read after a churn that would reuse a block freed too early.
import './export_own_class'

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new C(999));
    }
}

function main() {
    let t = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new H();
        h.c = new C(i % 10);
        h.add(1);
        h.add(2);
        churn();
        // a virtual call on an imported class may be to an override that drops anything under
        // `h`, so `h.cx` comes before the borrow of `h.c`; `sumOf` is listed as dropping nothing
        t += h.cx;
        const c = h.c;
        t += c.x + c.twice() + sumOf(h) + c.v.length;
    }

    assert(t == 45 * 100 * 4 + 1000 * 5);
    print("done.");
}
