// TSLANG_ISSUES 49 (-mm=own), reduced by powder_sandbox:
// - startA: a nullable fresh local moved into a field when it is not null. On the other way it is
//   null and holds nothing, so its release there frees nothing.
// - startB: the result of a method of an exported class, called through its vtable, is the caller's
//   when no method of its family says it borrows (an override may not say it either).
// - moveAside: one field moved into another, the source cleared right after.

class Level {
    name: string = "a";
}

export class App {
    level: Level | null = null;
    other: Level | null = null;

    read(n: string): Level | null {
        return n == "" ? null : new Level();
    }

    startA(n: string) {
        const l = this.read(n);
        if (l) {
            this.level = l;
        }
    }

    startA2(n: string) {
        const l = this.read(n);
        if (!l) {
            return;
        }

        this.level = l;
    }

    startB(n: string) {
        const l = this.read(n);
        this.level = l;
    }

    moveAside() {
        this.other = this.level;
        this.level = null;
    }
}

function main() {
    const a = new App();
    a.startA("");
    assert(a.level == null);
    a.startA("x");
    assert(a.level != null);
    a.startA2("");
    a.startA2("y");
    a.startB("x");
    a.moveAside();
    print(a.level != null, a.other != null);
    assert(a.level == null && a.other != null && a.other.name == "a");
    a.startB("");
    assert(a.level == null);
    print("done.");
}
