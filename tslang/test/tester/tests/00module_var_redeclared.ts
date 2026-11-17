// A module `var` declared again is the same variable: again with no value it is nothing, with one an
// assignment, run in order with the module's code ("redefinition of symbol named 'v'" before).
var v = 1;
var w = v + 1;
var v = 5;
var u: number;
var u: number;

// parameters are names in the types after them (`b: typeof a`, `): typeof a`), as in their defaults
function sum(a: number, b: typeof a) { return a + b; }
function bang(a: string): typeof a { return a + "!"; }
function wrap(a: number): { a: typeof a } { return { a }; }

function main() {
    assert(v == 5 && w == 2);
    u = 3;
    assert(u == 3);
    assert(sum(1, 2) == 3 && bang("x") == "x!" && wrap(5).a == 5);
    print("done.");
}
