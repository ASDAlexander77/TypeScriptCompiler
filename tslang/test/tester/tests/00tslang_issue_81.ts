// TSLANG 81: a value moved on some paths only (`const c = new Ctl(); if (c.read()) this.lights.push(c);`)
// is released on the edges that did not move it.
@final
class Ctl {
    node: int = 0;
    read(ok: boolean): boolean { return ok; }
}
@final
class Model {
    lights: Ctl[] = [];
    load(n: int) {
        for (let i = 0; i < n; i++) {
            const c = new Ctl();
            if (c.read(i != 2)) {
                this.lights.push(c);
            }
        }
    }
    load2(n: int) {
        const c = new Ctl();
        if (c.read(n != 2)) {
            this.lights.push(c);
        }
    }
}
// moved into a field on one way only (was own_err_move_some_paths_field)
class C {
    x: number = 5;
}

class H {
    c: C = new C();
}

function intoField(argc: number): number {
    const h = new H();
    const a = new C();
    a.x = 7;
    if (argc > 5) {
        h.c = a;
    }

    return h.c.x;
}

// moved where a second test passes, not only where the value is set (was own_err_moved_when_tested_other)
class Level {
    name: string = "a";
}

class App {
    level: Level | null = null;

    read(n: string): Level | null {
        return n == "" ? null : new Level();
    }

    start(n: string) {
        const l = this.read(n);
        if (l && n == "y") {
            this.level = l;
        }
    }
}

function main() {
    const m = new Model();
    m.load(4);
    m.load2(1);
    m.load2(2);
    assert(m.lights.length == 4, "lights");
    assert(intoField(1) == 5 && intoField(9) == 7, "field");
    const app = new App();
    app.start("x");
    assert(app.level == null, "x");
    app.start("y");
    assert(app.level != null, "y");
    print("done.");
}
