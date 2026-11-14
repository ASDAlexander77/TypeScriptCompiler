// TSLANG 80: a parameter a constructor stores into a field, then reads in a loop that calls
// (`constructor(public name: string)`, then `${name}`): `this` never escapes the constructor, so no
// call reaches the object, and the read takes the field.
@final
class Ctl {
    node: int = 0;
    read(ok: boolean): boolean { return ok; }
}
@final
class Model {
    lights: Ctl[] = [];
    constructor(public name: string, n: int) {
        for (let i = 0; i < n; i++) {
            if (i == 1) {
                print(`skip ${name}`);
                continue;
            }
            const c = new Ctl();
            c.node = i;
            this.lights.push(c);
        }
    }
}
function main() {
    const m = new Model("m", 4);
    assert(m.lights.length == 3 && m.name == "m", "model");
    print("done.");
}
