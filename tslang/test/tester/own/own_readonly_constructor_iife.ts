// An arrow the constructor calls where it is written runs inside the constructor: it may write the class's
// readonly fields, as tsc has it (readonlyMembers.ts), nested ones too. Every memory model.
class C {
    readonly a: number;
    readonly b = 1;
    constructor(n: number) {
        (() => { this.a = n; this.b = n + 1; })();
        (() => { (() => { this.b += 10; })(); })();
    }
}
function main() { const c = new C(2); print(c.a, c.b); assert(c.a == 2 && c.b == 13); print("done."); }
