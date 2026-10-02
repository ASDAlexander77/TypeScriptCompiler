// -mm=own, phase 7 rejects: `o.make` is assigned another function, so the call through the field
// is not known to be `make`'s, and may overwrite the field it is read from.
class C {
    constructor(public x: number) {}
}

function main() {
    const o = {
        make(): C {
            return new C(1);
        },
    };
    o.make = function (): C {
        return new C(2);
    };
    let c = o.make();
    print(c.x);
    c = o.make();
    print(c.x);
}
