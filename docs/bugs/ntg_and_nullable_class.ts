// `boolean && (class | null)` as an if condition fails to compile: "can't instantiate function.
// '___cast' not all generic types can be identified". Expected output: 7, done.
class Handle {
    value: number;
    constructor() {
        this.value = 7;
    }
}

function use(enabled: boolean, handle: Handle | null): number {
    if (enabled && handle) {
        return handle.value;
    }
    return 0;
}

const v = use(true, new Handle());
print(v);
assert(v == 7);
print("done");
