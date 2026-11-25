// TSLANG 191: a class's static initializers and static blocks run after the module's globals declared before the
// class, as in TypeScript: `const x = pick(1); class C { static y = x; }` gave C.y null (the globals' own start-up
// constructors ran after every class's).
function pick(n: number): string {
    return "m" + n;
}

const x = pick(1);

class C {
    static y: string = x;
    static z: string;
    static {
        C.z = x + "!";
    }
}

const after = C.y + "?";

function main() {
    assert(C.y == "m1", "a static initializer reads a global before the class");
    assert(C.z == "m1!", "a static block reads a global before the class");
    assert(after == "m1?", "a global after the class reads its static");
    print("done.");
}
