// A bound method given as a function type with more parameters after its own ignores the rest, as JavaScript
// does: `o.something ?? val.something` where `o.something` takes `(b?: string)` and the method none, a class's
// method given for `(a, b) => void`. It goes into a box with an adapter bound to it; made inside a branch, the
// adapter is what the merge holds (under rc it was released at the end of the branch, and the call read freed
// memory).
interface ReturnVal {
    something(): void;
}

class C {
    n = 0;
    something() {
        this.n++;
    }
}

let calls = 0;
let seen = "";

function mk(): ReturnVal {
    return {
        something() {
            calls++;
        }
    };
}

function run(options: { something?(b?: string): void }, val: ReturnVal) {
    const something = options.something ?? val.something;
    something("x");
}

function run2(options: { something?(b?: string): void }, c: C) {
    const something = options.something ?? c.something;
    something("y");
}

function apply(f: (a: number, b: number) => void) {
    f(1, 2);
}

function main() {
    const v = mk();
    const c = new C();
    for (let i = 0; i < 100; i++) {
        run({}, v);
        run2({}, c);
        run2({}, new C());
    }

    assert(calls == 100);
    assert(c.n == 100);

    run({ something(b?: string) { seen = b; } }, v);
    assert(seen == "x");

    apply(c.something);
    assert(c.n == 101);

    print("done.");
}
