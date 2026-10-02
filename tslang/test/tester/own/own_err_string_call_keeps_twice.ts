// -mm=own rejects, and does not copy: one string given twice to a call that keeps both arguments.
// The retains have no single use each, so no copy is made.
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
