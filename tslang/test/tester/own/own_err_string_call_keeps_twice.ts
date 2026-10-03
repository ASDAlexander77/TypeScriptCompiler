// -mm=own rejects, and does not copy: a field's string given twice to a call that keeps both
// arguments. A keeping call's retains are in the callee, so `main` has no retain to turn into a
// copy: the call takes the value without one (spec 22.4).
class P {
    a = "";
    b = "";
}

const p = new P();

function both(x: string, y: string) {
    p.a = x;
    p.b = y;
}

function main() {
    const h = new P();
    h.a = "s" + 1;
    both(h.a, h.a);
    print(p.a, p.b);
}
