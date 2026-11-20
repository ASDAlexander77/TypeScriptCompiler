// A parameter's destructuring pattern that nothing gives types to has `any` elements, as TypeScript's implicit any:
// `function f({ a, b })`, `function g([x, y])`, `({ a }) => a`, a skipped element (`[, z]`) holding its place. It was
// "can't resolve type for binding pattern 'a', provide default initializer".
function f({ a, b }) { return b; }
function g([x, y]) { return y; }
function skip([, z]) { return z; }
class C { m({ x }) { return x; } }

function main() {
    assert(f({ a: 1, b: "two" }) == "two");
    assert(g([1, 2]) == 2);
    assert(skip(["a", "b"]) == "b");
    assert(new C().m({ x: 3 }) == 3);
    const ar = ({ a, b }) => a + b;
    assert(ar({ a: 1, b: 2 }) == 3);

    let n: number = 0;
    for (let i = 0; i < 1000; i++) {
        const s = f({ a: i, b: "s" + i });
        const p = g(["p" + i, i]);
        n += p;
    }
    assert(n == 499500);
    print("done.");
}
