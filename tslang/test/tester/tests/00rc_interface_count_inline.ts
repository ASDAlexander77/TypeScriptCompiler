// Under rc, an interface value is retained and released through the type descriptor its vtable names,
// and for a class or object literal instance - a block counted the plain way - the count is now
// changed inline: one more reference on a retain, one less on a release that is not the last. The
// last reference, a shared or an immortal block, and any other kind still go through the
// descriptor's routine (OwnershipRoutineLogic::releaseViaInterfaceTag).
//
// Each value below is read after allocations of the same size have reused anything freed too early,
// so a release that dropped one reference too many reads back as a wrong number, and one that
// dropped too few leaks (the loops run long enough for that to show as memory, not as a value).
interface Shape {
    area(): number;
    id: number;
}

class Square implements Shape {
    data: number[] = [0, 0, 0];
    constructor(public id: number, public side: number) {}
    area() { return this.side * this.side; }
}

@value
interface Hit {
    readonly shape: Shape;
    readonly dist: number;
}

function churn(n: number) {
    let total = 0;
    for (let i = 0; i < n; i++) {
        const s = new Square(-1, i);
        total += s.side;
    }
    return total;
}

// a hit holding the only reference to a fresh instance: copied, then the copies dropped one by one
function makeHit(i: number): Hit | null {
    if (i % 3 == 0) {
        return null;
    }

    return { shape: new Square(i, i % 7), dist: i };
}

function closest(n: number): Hit | undefined {
    let best: Hit | undefined = undefined;
    let bestDist = +Infinity;
    for (let i = 0; i < n; i++) {
        const h = makeHit(n - i);
        if (h != null && h.dist < bestDist) {
            best = h;
            bestDist = h.dist;
        }
    }

    return best;
}

// an object literal behind the interface, and two interface values over one instance
function literals() {
    const lit: Shape = { id: 77, area: () => 5 };
    let a: Shape = lit;
    let b: Shape = a;
    churn(100);
    assert(a.id == 77 && b.area() == 5, "literal");
    a = new Square(78, 3);
    churn(100);
    assert(b.id == 77 && a.area() == 9, "literal after reassign");
}

// a global's instance is immortal: retained and released through the interface, it must stay
const globalSquare = new Square(500, 4);

function globals() {
    for (let i = 0; i < 1000; i++) {
        const s: Shape = globalSquare;
        const h: Hit = { shape: s, dist: i };
        assert(h.shape.id == 500, "global");
    }
}

function main() {
    let sum = 0;
    for (let r = 0; r < 2000; r++) {
        const h = closest(10);
        churn(10);
        if (h !== undefined) {
            sum += h.shape.id + h.shape.area();
        }
    }

    // closest(10): the smallest dist with i % 3 != 0 is 1 (Square(1, 1), area 1)
    assert(sum == 2000 * 2, "closest");

    // a hit kept while the loop's own copies are dropped
    const kept = closest(20);
    churn(1000);
    assert(kept !== undefined && kept.shape.id == 1 && kept.shape.area() == 1, "kept");

    literals();
    globals();

    print("done.");
}
