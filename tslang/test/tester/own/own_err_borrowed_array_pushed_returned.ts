// TSLANG_ISSUES 62: an array changed through a local that borrows the field holding it is still the
// field's: the local cannot be returned.
class V {
    n = 1;
}

class H {
    arr: V[] = [];
}

export function f(h: H): V[] {
    const a = h.arr;
    a.push(new V());
    return a;
}

function main() { print(f(new H()).length); }
