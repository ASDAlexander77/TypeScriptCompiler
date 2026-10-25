// A module-level array that is only ever pushed onto - a keeper - keeps what it holds to the end of
// the program, so under -mm=own a value pushed into it is made immortal where it is known what it
// relies on (docs/superpowers/specs/2026-10-25-own-immortal-keepers-design.md): a bound method's
// object, a closure's box and the cells it borrows, an object pushed directly. The owners' releases
// then free nothing, and the callbacks still run after the functions that registered them return
// and the memory they released is handed out again.
type Callback = (n: number) => void;
let retained: Callback[] = [];

class Registry {
    register(handler: Callback): void {
        retained.push(handler);
        handler(0);                          // still the caller's to use after the push
    }
}

class Counter {
    count: number[] = [0.0];
    constructor(registry: Registry) {
        registry.register(this.onTick);      // the object is main's, and kept for good
    }
    onTick(n: number): void {
        this.count[0] += n;
    }
}

// a parameter passed on: the caller makes it immortal
function forward(r: Registry, h: Callback): void {
    r.register(h);
}

// said, as a function another module may call has to
@immortalizes("h")
export function forwardExported(h: Callback): void {
    new Registry().register(h);
}

class Node {
    constructor(public v: number) {}
}

let nodes: Node[] = [];

let sums: number[] = [0.0, 0.0, 0.0, 0.0];

function setupBound(): Counter {
    return new Counter(new Registry());
}

function setupClosure(): void {
    const total = [100.0];
    new Registry().register((n: number) => { total[0] += n; sums[0] = total[0]; });
    total[0] += 1.0;                         // the frame still uses what the closure captured
}

function setupForward(): void {
    const box = [7.0];
    forward(new Registry(), (n: number) => { box[0] += n; sums[1] = box[0]; });
}

function setupExported(): void {
    const box = [20.0];
    forwardExported((n: number) => { box[0] += n; sums[2] = box[0]; });
}

function setupNodes(): void {
    nodes.push(new Node(5));
    const local = new Node(6);
    nodes.push(local);                       // and still the local's
    sums[3] = local.v;
}

class Holder {
    first: Callback;
    constructor() { this.first = retained[0]; }   // a keeper's element goes anywhere
}

function churn(): number {
    let sum = 0.0;
    for (let i = 0; i < 20000; i++) {
        const a = [i * 1.0, 2.0, 3.0];
        sum += a[0];
    }
    return sum;
}

function main() {
    const counter = setupBound();
    setupClosure();
    setupForward();
    setupExported();
    setupNodes();
    churn();
    for (let k = 0; k < 3; k++) {
        for (const cb of retained) {
            cb(1);
        }

        churn();
    }

    const holder = new Holder();
    holder.first(10);
    print(counter.count[0], sums[0], sums[1], sums[2], nodes[0].v + nodes[1].v, retained.length);
    assert(counter.count[0] == 13 && sums[0] == 104 && sums[1] == 10 && sums[2] == 23);
    assert(nodes[0].v + nodes[1].v == 11 && sums[3] == 6 && retained.length == 4);
    print("done.");
}
