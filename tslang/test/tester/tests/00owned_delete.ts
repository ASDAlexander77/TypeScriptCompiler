// Under reference counting `delete` gives up a reference rather than freeing outright. The
// reference it gives up is the one the expression named, so whatever else would have released
// that same reference - the end-of-block release of a temporary nobody claimed, or the storage
// a variable lives in - has to stop doing so, or the block is let go of twice.
//
// `deleteFromAStaticMethod` is the case that carries this: with the fix reverted it faults three
// runs in three, and the others double-free without anything noticing. They are here as shape
// coverage - a variable with storage, a loop, an object owning a string, a delete on one branch
// of two - not because any of them discriminates today.
//
// What `delete` means differs by model, so a second reference to a deleted object is not
// something a test shared by all three can say anything about: under `gc` and `none` the block
// is freed outright and any other reference dangles, while under `rc` it is one owner going
// away. Every case below holds exactly one.

let glb = 0;

function churn(): number {
    let n = 0;
    for (let i = 0; i < 400; i++) {
        const s = "filler " + i;
        n = n + s.length;
    }

    return n;
}

class Cell {
    value: number = 0;
    label: string = "";
}

// No initialisers, so no constructor runs and nothing takes the reference `new` hands back on
// the way in. That is the shape `00class_static.ts` is built from, and the one where the
// end-of-block release of an unclaimed temporary is what runs the second time.
class Bare {
    value: number;
}

function deleteALet() {
    let c = new Cell();
    c.value = 8;
    print(c.value);
    delete c;
    churn();
    assert(glb == 0, "a variable with storage, deleted");
}

// The shape `00class_static.ts` is built from, and the one that faults rather than merely
// double-freeing quietly: a static method that builds an instance, prints twice - each print of
// a number builds a string the block also releases - and deletes it.
class Static {
    static count: number;
    pin: number;

    static run() {
        const s = new Static();
        s.pin = 10;
        print(s.pin);

        Static.count = 20;
        print(Static.count);

        delete s;
    }
}

function deleteFromAStaticMethod() {
    Static.run();
    churn();
    assert(Static.count == 20, "the static method returned with its heap intact");
}

function deleteManyInALoop() {
    let seen = 0;
    for (let i = 0; i < 200; i++) {
        const b = new Bare();
        b.value = i;
        seen = seen + b.value;
        delete b;
    }

    churn();
    assert(seen == 19900, "two hundred allocations, each deleted");
}

function deleteAnObjectHoldingAString() {
    const c = new Cell();
    c.label = "held" + glb;
    assert(c.label == "held0", "the field holds what was built");
    delete c;
    churn();
    assert(glb == 0, "and the string went with it");
}

function deleteInsideABranch() {
    let total = 0;
    for (let i = 0; i < 100; i++) {
        const c = new Cell();
        c.value = i;
        if (i % 2 == 0) {
            total = total + c.value;
            delete c;
        } else {
            total = total + c.value;
        }
    }

    churn();
    assert(total == 4950, "deleted on one path and not the other");
}

function main() {
    deleteFromAStaticMethod();
    deleteALet();
    deleteManyInALoop();
    deleteAnObjectHoldingAString();
    deleteInsideABranch();

    print("done.");
}
