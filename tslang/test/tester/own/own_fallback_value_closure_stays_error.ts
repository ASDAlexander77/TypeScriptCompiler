// -mm=own --own-fallback=count: a @value value stored in two holders would be counted, and its count
// is an extra owner of the object it holds. That object has a closure field, which a take out of a
// counted object cannot give an owner (OwnershipRoutineLogic::canCountRetainOf), so no counted value
// may reach it: the record is asked what its class fields may hold, and stays an error.
class C {
    f: () => number = () => 1;
    constructor(public v: number) {}
}

@value
class V {
    constructor(public readonly c: C) {}
}

class Holder {
    t: V | null = null;
}

function link(h: Holder, g: Holder, t: V) {
    h.t = t;
    g.t = t;
}

function main() {
    const h = new Holder();
    const g = new Holder();
    link(h, g, new V(new C(5)));
    print(h.t!.c.v + g.t!.c.v);
    print("done.");
}
