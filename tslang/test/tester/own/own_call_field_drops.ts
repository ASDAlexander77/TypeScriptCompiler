// -mm=own: a callee that destroys only the old values of the places it overwrites drops a borrow
// only as those stores would in the caller (OWN_DROP_FIELDS_ATTR_NAME). A method that replaces an
// array field of its own object, called on an object held in a field of `this` - a camera kept by
// a render pass - leaves the borrow of that object alone, directly, through a method that calls it,
// and through an interface. The same overwrite of a place on the chain still drops it
// (own_err_call_field_drops_owner, own_err_call_field_drops_under).
class Cam {
    position: number[] = [0.0, 0.0];
    private initial: number[] = [0.0, 0.0];

    init(x: number, y: number): void {
        this.initial = [x, y];
        this.reset();
    }

    reset(): void {
        this.position = [this.initial[0], this.initial[1]];
    }
}

interface Mover {
    move(c: Cam): void;
}

class Step implements Mover {
    move(c: Cam): void {
        c.position = [c.position[0] + 1.0, c.position[1]];
    }
}

function churn() {
    const keep: Cam[] = [];
    for (let i = 0; i < 2000; i++) {
        const c = new Cam();
        c.position = [77777.0, 77777.0];
        keep.push(c);
    }

    return keep.length;
}

class Pass {
    private camera: Cam = new Cam();

    direct(): number {
        this.camera.reset();
        churn();
        return this.camera.position[0];
    }

    nested(): number {
        this.camera.init(3.0, 4.0);
        churn();
        return this.camera.position[0] + this.camera.position[1];
    }

    viaInterface(m: Mover): number {
        m.move(this.camera);
        churn();
        return this.camera.position[0];
    }
}

function main() {
    const p = new Pass();
    assert(p.direct() == 0, "a method that replaces its own field");
    assert(p.nested() == 7, "through a method that calls it");
    assert(p.viaInterface(new Step()) == 4, "through an interface");
    print(p.direct(), p.nested(), p.viaInterface(new Step()));   // 3 7 4 (init kept 3, 4)
    print("done.");
}
