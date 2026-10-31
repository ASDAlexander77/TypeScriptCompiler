// A @cpp class that may be null in a union with a boolean (`h && n == 3`) is tested for truth by
// its pointer alone: its objects are C++ objects, and `instanceof` can't ask them what they are.
// The "C++ object" is made by hand, as in tests/00cppvtable_class.ts.

@cpp
class Handle {
    value: s32;
    @slot(0) first(): s32;
}

@cpp
class Other {
    @slot(0) first(): s32;
}

type ObjectData = [vtable: Opaque, value: s32];

function slot0(self: Opaque): s32 {
    return 10;
}

function both(h: Handle | null, n: number): boolean {
    if (h && n == 3) {
        return true;
    }

    return false;
}

function either(enabled: boolean, h: Handle | null): boolean {
    return enabled && h ? true : false;
}

function main() {
    let vtable: Opaque[] = [<Opaque>slot0];
    let data: ObjectData = [<Opaque>Ref(vtable[0]), 32];
    const object = <Opaque>Ref(data);

    const handle = <Handle>object;
    const none: Handle | null = null;

    assert(both(handle, 3), "a handle and true");
    assert(!both(handle, 4), "a handle and false");
    assert(!both(none, 3), "null and true");

    assert(either(true, handle), "true and a handle");
    assert(!either(false, handle), "false and a handle");
    assert(!either(true, none), "true and null");

    // two @cpp classes and a boolean: neither class needs telling apart to be tested for truth
    const other = <Other>object;
    const flag = true;
    let a: Handle | Other | boolean | null = flag && other;
    assert(a ? true : false, "a @cpp object of either class");
    a = null;
    assert(!(a ? true : false), "null");

    print("done.");
}
