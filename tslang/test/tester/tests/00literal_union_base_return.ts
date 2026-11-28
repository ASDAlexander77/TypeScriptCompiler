// regression test: a literal beside its own base type in a union is that type.
//
// bug found and fixed: a function returning `c ? 1 : 0` on one path and an `s32` on another got the
// inferred return type `0 | 1 | s32`, and was refused as a `=> s32` ("return type '0 | 1 | s32' is
// not 's32'"), as a method given as a callback was.

class Keys {
    pressed = 0;

    onKey(key: number, action: number) {
        if (action != 1) {
            return key == 2 ? 1 : 0;
        }
        if (key == 3) {
            return 0;
        }
        this.pressed++;
        return 1;
    }
}

function pick(k: number) {
    if (k > 2) return k == 3 ? 1 : 0;
    return 1;
}

function same(k: s32) {
    if (k > 2) return k == 3 ? 1 : 0;
    return k;
}

function callKey(cb: (key: number, action: number) => s32, key: number, action: number) {
    return cb(key, action);
}

function main() {
    const keys = new Keys();
    const cb: (key: number, action: number) => s32 = keys.onKey;
    assert(cb(2, 0) == 1);
    assert(cb(3, 1) == 0);
    assert(cb(4, 1) == 1);
    assert(keys.pressed == 1);
    assert(callKey(keys.onKey, 5, 1) == 1);
    assert(keys.pressed == 2);

    const cf: (k: number) => s32 = pick;
    assert(cf(3) == 1);
    assert(cf(4) == 0);
    assert(cf(1) == 1);

    const cs: (k: s32) => s32 = same;
    assert(cs(3) == 1);
    assert(cs(2) == 2);

    print("done.");
}
