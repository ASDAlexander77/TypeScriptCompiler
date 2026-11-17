// -mm=own: `return s` inside a `using` block. The dispose call comes between the read of `s` and the
// return, so an exception there unwinds to the block's handler with `s` not yet given up: the handler
// still frees it, and the normal path moves it out. Also when the constructor throws (before the read)
// and when dispose throws (after it).
class Res {
    constructor(public n: string) {
        if (n == "bad") throw "ctor";
    }

    [Symbol.dispose]() {
        if (this.n == "boom") throw "dispose";
        disposed++;
    }
}

let disposed = 0;

function returnFromBlock(pre: string, r: string) {
    let s = pre + "1";
    {
        using a = new Res(r);
        return s;
    }
}

function nested(pre: string) {
    let s = pre + "2";
    {
        using a = new Res("a");
        {
            using b = new Res("b");
            return s;
        }
    }
}

function main() {
    let total = 0;
    for (let i = 0; i < 1000; i++) {
        total += returnFromBlock("x" + i, "ok").length;
        total += nested("y" + i).length;
        if (i % 10 == 0) {
            try {
                returnFromBlock("z" + i, "boom");
            } catch (e) {
                total++;
            }

            try {
                returnFromBlock("w" + i, "bad");
            } catch (e) {
                total++;
            }
        }
    }

    assert(returnFromBlock("p", "ok") == "p1");
    assert(nested("q") == "q2");
    assert(total == 9780 + 200, "lengths and catches");
    assert(disposed == 3003, "every dispose ran");
    print("done.");
}
