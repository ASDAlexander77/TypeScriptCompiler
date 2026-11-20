// A function declared in a block of the module's top-level code reads the block's locals, as one declared in a
// block of a function does: what it captures was found by the pass over the module's declarations, which does
// not know them, and it read them across its isolated region ("'ts.Load' op using value defined outside the
// region"); called, it crashed the compiler. So did a method of an object literal there.
let total = 0;
{
    let n = 3;
    function fact(k: number): number {
        return k <= 1 ? 1 : k * fact(k - 1);
    }
    function bump() {
        n = n + 1;
    }
    bump();
    total = fact(n);
}

const fs: (() => number)[] = [];
for (let i = 0; i < 3; i++) {
    function get(): number {
        return i * 10;
    }
    fs.push(get);
}

if (total > 0) {
    let m = 5;
    function early() {
        return m;
    }
    total += early();
}

let fromMethod = 0;
{
    let x = 7;
    const o = {
        foo() {
            return x;
        },
    };
    fromMethod = o.foo();
}

function main() {
    assert(total == 24 + 5);
    assert(fs[0]() == 0 && fs[2]() == 20);
    assert(fromMethod == 7);
    print("done.");
}
