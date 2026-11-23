// -mm=own --own-fallback=count: a tuple stored in two holders' `[C, number] | null` fields would be
// counted, but each kept retain is of the field's type: a tagged union, whose payload under own has
// nothing to give an owner through (OwnershipRoutineLogic::canCountRetainOf). Its group is refused
// for the retains' type, not only its members' (the tuple's), so it stays own's error at its place
// rather than failing in lowering.
class C {
    constructor(public v: number) {}
}

class Holder {
    t: [C, number] | null = null;
}

function link(h: Holder, g: Holder, t: [C, number]) {
    h.t = t;
    g.t = t;
}

function main() {
    const h = new Holder();
    const g = new Holder();
    const c = new C(5);
    const t: [C, number] = [c, 1];
    link(h, g, t);
    print(h.t![0].v + g.t![0].v);
    print("done.");
}
