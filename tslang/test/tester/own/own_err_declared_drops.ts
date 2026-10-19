// -mm=own rejects: `strlen` is declared without `@nodrops`, so the method that calls it may
// destroy anything, and `this.value` cannot be lent to it (own_declared_no_drops).
declare function strlen(s: string): index;

namespace __S {
    export function size(this: string): int {
        return <int>strlen(this);
    }
}

class W {
    constructor(private value: string) {}

    size(): int {
        return this.value.size();
    }
}

function main() {
    const w = new W("abc");
    print(w.size());
}
