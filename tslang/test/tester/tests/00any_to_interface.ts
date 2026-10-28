// An `any` holds the kind of what it holds, not which interface an interface value was made for.
// `<Holder><any>runner` read the Runner value's vtable as Holder's and called garbage; a class object
// in the box was read as an interface's (this, vtable) pair. The object is now asked what class it
// is: a class with a vtable for the interface gives it, any other class of the module throws, and
// anything else is read as before, so a value boxed as the interface it is cast to round-trips.

class Item {
    constructor(public n: number) {}
}

interface Runner {
    walk(a: Item[]): number;
}

interface Holder {
    f: (v: Item) => void;
}

// a Runner, and no Holder anywhere in the module
class Walker implements Runner {
    constructor(public f: (v: Item) => void) {}
    walk(a: Item[]): number {
        return 1;
    }
}

class Keeper implements Holder {
    constructor(public f: (v: Item) => void) {}
}

// a Holder through a cast made only further down the module (sawLater)
class Later {
    constructor(public f: (v: Item) => void) {}
}

let shown: number = 0;
function show(v: Item) {
    shown += v.n;
}

function otherInterface() {
    let caught = false;
    try {
        const r: Runner = new Walker(show);
        const h = <Holder><any>r;
        h.f(new Item(1000));
    } catch (e) {
        caught = true;
    }

    assert(caught, "a Runner over a Walker is no Holder");
}

function classObject() {
    let caught = false;
    try {
        const h = <Holder><any>new Walker(show);
        h.f(new Item(1000));
    } catch (e) {
        caught = true;
    }

    assert(caught, "a Walker is no Holder");
}

function implementer() {
    const h1 = <Holder><any>new Keeper(show);
    h1.f(new Item(1));

    const kh: Holder = new Keeper(show);
    const h2 = <Holder><any>kh;
    h2.f(new Item(2));
}

function castBeforeTheVTable() {
    const h = <Holder><any>new Later(show);
    h.f(new Item(4));
}

function roundTrip() {
    const r: Runner = new Walker(show);
    const back = <Runner><any>r;
    assert(back.walk([]) == 1, "round trip");
}

function sawLater() {
    const h: Holder = new Later(show);
    h.f(new Item(8));
}

function main() {
    otherInterface();
    classObject();
    implementer();
    castBeforeTheVTable();
    roundTrip();
    sawLater();
    assert(shown == 15, "every Holder reached show");
    print("done.");
}
