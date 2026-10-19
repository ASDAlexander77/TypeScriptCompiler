// -mm=own: a C function declared `@nodrops` destroys nothing a TypeScript caller can reach, so a
// borrow of a field survives a call that reaches it, as it survives a call to a function defined
// here that destroys nothing. Without the mark the declaration may destroy anything
// (own_err_declared_drops).
@nodrops declare function strlen(s: string): index;

namespace __S {
    export function size(this: string, extra: string): int {
        return <int>strlen(this) + <int>strlen(extra);
    }
}

class W {
    constructor(private value: string) {}

    size(extra: string): int {
        return this.value.size(extra);
    }
}

class Pad {
    constructor(public s: string) {}
}

function churn() {
    const junk: Pad[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new Pad("junk" + i));
    }
}

function main() {
    let t = 0;
    for (let i = 0; i < 10000; i++) {
        const w = new W("abc" + (i % 10));
        churn();
        t += w.size("de");
    }

    assert(t == 60000);
    print("done.");
}
