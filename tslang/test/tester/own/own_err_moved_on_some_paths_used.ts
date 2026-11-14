// -mm=own rejects: `c`, pushed only where `read` succeeds, is released on the other way (TSLANG 81),
// so a use of it after the ways join may find it moved.
@final
class Ctl {
    node: int = 0;

    read(ok: boolean): boolean {
        return ok;
    }
}

@final
class Model {
    lights: Ctl[] = [];

    load(n: int) {
        const c = new Ctl();
        if (c.read(n != 2)) {
            this.lights.push(c);
        }
        print(c.node);
    }
}

function main() {
    new Model().load(1);
}
