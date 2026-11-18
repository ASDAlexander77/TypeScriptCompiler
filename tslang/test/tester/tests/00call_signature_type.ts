// `{ (n: number): string }`, one call signature and nothing else, is the function type `(n: number) => string`
namespace M {
    export class C {
        m(fn: { (n: number): string; }, n2: number): string {
            return fn(n2);
        }
    }
}

type Handler = { (a: number, b: number): number };

function apply(h: Handler) {
    return h(2, 3);
}

function main() {
    const c = new M.C();
    assert(c.m(function (n) { return "hello: " + n; }, 18) == "hello: 18", "call signature parameter");
    assert(apply((a: number, b: number) => a * b) == 6, "call signature alias");

    const handlers: Handler[] = [(a: number, b: number) => a + b];
    assert(handlers[0](1, 2) == 3, "element of call signature type");

    print("done.");
}
