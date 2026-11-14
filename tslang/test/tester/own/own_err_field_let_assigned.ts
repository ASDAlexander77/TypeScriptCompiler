// -mm=own, phase 3 rejects: `a` borrows `h.a`, and is changed in place after it is given an array of its
// own (`a.push`, through its slot). A `let` that is only read after a fresh value (TSLANG 66) borrows it
// from a slot of its own instead; one changed through its slot is not split, so it cannot be assigned.
class H {
    a: number[] = [1.0];
}

function main() {
    const h = new H();
    let a = h.a;
    print(a.length);
    a = [2.0, 3.0];
    a.push(4.0);
    print(a.length);
}
