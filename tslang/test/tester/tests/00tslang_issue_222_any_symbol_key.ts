// TSLANG 222: `p[sym]` of an `any` holding `{ [sym]: 6 }` read `undefined`: the key was the symbol's text
// (`Symbol(k)`), not the name the `[sym]` field is laid out by. (-mm=own refuses `x[k]` of `any`.)

const sym = Symbol("k");
const other = Symbol("k");

function main() {
    const p: any = { [sym]: 6, a: 1 };
    assert(p[sym] == 6, "symbol key of an any");
    assert(p[other] === undefined, "another symbol of the same description");
    assert(p["a"] == 1, "string key");
    print("done.");
}
