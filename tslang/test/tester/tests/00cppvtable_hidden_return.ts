// A @cppvtable method declared to return a tuple returns a C++ record (a struct, a class such as
// RefCountPtr<T>) by value. MSVC returns a record from an instance method through a hidden pointer,
// passed after `this` and returned in turn, so the slot is called as fn(this, &result, args) and the
// call's value is what fn wrote there. The "C++ methods" here are written in that shape.

declare function memcpy(dst: Opaque, src: Opaque, count: index): Opaque;

type Pair = [first: s32, second: s32];        // 8 bytes: still a hidden pointer, as an instance method's
type Triple = [a: u64, b: u64, c: u64];       // 24 bytes

@cppvtable
class Methods {
    value: s32;
    pair(x: s32): Pair;                         // slot 0
    triple(): Triple;                           // slot 1
    plain(x: s32, y: s32): s32;                 // slot 2: a number, returned as before
}

type ObjectData = [vtable: Opaque, value: s32];

function slotPair(self: Opaque, result: Ref<Pair>, x: s32): Ref<Pair> {
    let value: Pair = [x + Deref(<Ref<ObjectData>>self).value, x * 2];
    memcpy(<Opaque>result, <Opaque>Ref(value), sizeof<Pair>());
    return result;
}

function slotTriple(self: Opaque, result: Ref<Triple>): Ref<Triple> {
    let value: Triple = [1, 2, 0x123456789];
    memcpy(<Opaque>result, <Opaque>Ref(value), sizeof<Triple>());
    return result;
}

function slotPlain(self: Opaque, x: s32, y: s32): s32 {
    return x * y;
}

function main() {
    let vtable: Opaque[] = [<Opaque>slotPair, <Opaque>slotTriple, <Opaque>slotPlain];
    let data: ObjectData = [<Opaque>Ref(vtable[0]), 100];
    const methods = <Methods><Opaque>Ref(data);

    const pair = methods.pair(5);
    assert(pair.first == 105 && pair.second == 10, "a record returned through the hidden pointer");

    const triple = methods.triple();
    assert(triple.a == 1 && triple.b == 2 && triple.c == 0x123456789, "a bigger record");

    assert(methods.plain(6, 7) == 42, "a number is still returned in a register, after them");
    assert(methods.pair(1).second == 2, "a record read from the call's value");

    print("done.");
}
