// The methods of a C++ object made by MSVC (abi_host.cpp): one destructor slot, and every record
// returned through a pointer after `this`.

type Ptr = [p: Opaque | null];
type Obj = [p: Opaque | null];
type Two = [a: u64, b: Opaque | null];
// records passed by value and returned (abi_host.cpp's S4 ... R12)
type S4 = [a: u16, b: u8, c: u8];
type S8 = [a: u32, b: u32];
type S12 = [a: u32, b: u32, c: u32];
type S16 = [a: u64, b: u32, c: u32];
type S24 = [a: u64, b: u64, c: u64];
type Flags = [a: boolean, b: boolean, c: boolean];
type R12 = [a: u32, b: u32, c: u32];

declare function abi_object(): Opaque;
declare function abi_is_storage(p: Opaque | null, index: s32): s32;
declare function abi_heap_object(): Opaque;
declare function abi_delete_object(methods: Opaque): void;
declare function abi_guard_intact(methods: Opaque): s32;

@cpp
class Methods {
    value: s32;
    @slot(0) destructor(): void;  // the scalar deleting destructor
    @cppnontrivial
    @slot(1) make(x: s32): Ptr;
    @slot(2) obj(x: s32): Obj;
    @slot(3) two(x: s32): Two;
    @slot(4) plain(a: s32, b: s32): s32;
    @slot(5) narrow(b: boolean, u: u8, s: s8): s32;
    // C++ overloads, each with its slot: MSVC groups them in reverse declaration order
    @slot(8) over(x: s32): s32;
    @slot(7) over(x: f64): s32;
    @slot(6) over(a: s32, b: s32): s32;
    // records by value (as each ABI passes them), and returned
    @slot(9) byValue(k: s32, a: S4, b: S8, c: S12, d: S16, e: S24, tail: s32): u64;
    @slot(10) pair(d: S16, tail: s32): u64;
    @slot(11) flags(x: s32): Flags;
    @slot(12) r12(x: s32): R12;
    // a C++ non-virtual method (no @slot), called by its symbol, and its overload
    @linkname("?scaled@Methods@@QEAAHH@Z")
    scaled(x: s32): s32;
    @linkname("?scaled@Methods@@QEAAHN@Z")
    scaled(x: f64): s32;
}

// A C++ interface (abi_host.cpp's Callback) implemented here: C++ calls Counter's methods through the
// vtable tslang makes for it
@cpp
class Callback {
    @slot(0) destructor(): void;
    @slot(1) message(severity: u8, text: string): void;
    @slot(2) count(): s32;
    @slot(3) isEven(x: s32): boolean;
    // C++ overloads: MSVC groups them in reverse declaration order
    @slot(5) pick(x: s32): s32;
    @slot(4) pick(x: f64): s32;
}

@cpp
class Counter extends Callback {
    total: s32 = 0;
    unexpected = 0;

    constructor(public start: s32) {
        super();
    }

    message(severity: u8, text: string): void {
        this.total += severity;
        if (text != "two" && text != "three") {
            this.unexpected++;
        }
    }

    count(): s32 {
        return this.total + this.start;
    }

    isEven(x: s32): boolean {
        return x % 2 == 0;
    }

    // overloads, each implementing its slot: C++ picks one, and so does tslang
    @slot(5) pick(x: s32): s32 {
        return x + this.start;
    }

    @slot(4) pick(x: f64): s32 {
        return <s32>(x * 100);
    }

    // TypeScript's own (no @slot, overriding nothing): not in the vtable C++ calls through
    doubled(): s32 {
        return this.count() * 2;
    }
}

declare function abi_call_callback(callback: Opaque): s32;
declare function abi_callback_is_even(callback: Opaque, x: s32): s32;
declare function abi_callback_pick(callback: Opaque): s32;

// A @cpp reference copied into a field, an array and a local, each given up at the end: under rc
// and own none of them is counted, since the object has no count in front of it. The word before
// it is C++'s (abi_host.cpp's guard), and stays as C++ wrote it while the copies are held.
class Holder {
    constructor(public methods: Methods) {}
}

function copies(methods: Methods): s32 {
    const held = new Holder(methods);
    const list = [methods, methods];
    let other: Methods | null = methods;
    assert(abi_guard_intact(<Opaque>methods) == 1, "nothing written in the word before a C++ object");
    return held.methods.plain(2, 3) + list[1].plain(1, 1) + (other ? other.plain(0, 0) : 0);
}

// arguments made from a wider value: the register's upper bits aren't zero
function callNarrow(methods: Methods, big: s32): s32 {
    return methods.narrow(big > 0, <u8>big, <s8>(big - 0x1237));
}

function main() {
    const methods = <Methods>abi_object();
    assert(methods.value == 7, "the first field follows the vtable pointer");
    assert(abi_is_storage(methods.make(3).p, 3) == 1, "a record with a destructor");
    assert(abi_is_storage(methods.obj(4).p, 5) == 1, "a trivial 8-byte record");
    const two = methods.two(7);
    assert(two.a == 0x1234567890 + 7 && abi_is_storage(two.b, 9) == 1, "a trivial 16-byte record");
    assert(methods.plain(6, 7) == 49, "a number");
    assert(callNarrow(methods, 0x1234) == 1000 + 0x34 * 10 - 3, "narrow arguments, extended");
    assert(methods.scaled(3) == 307, "a non-virtual method, by its symbol");
    const half: number = 0.5;
    assert(methods.scaled(half) == 507, "its overload, picked by the argument");

    // C++ overloads: one name, the slot picked by the arguments
    const small: u8 = 4;
    assert(methods.over(5) == 1005, "over(int)");
    assert(methods.over(small) == 1004, "over(int), from a u8");
    assert(methods.over(2.5) == 2025, "over(double)");
    assert(methods.over(half) == 2005, "over(double), from a number");
    assert(methods.over(6, 7) == 3042, "over(int, int)");

    const counter = new Counter(5);
    assert(abi_call_callback(<Opaque>counter) == 10 && counter.unexpected == 0, "C++ calls the methods implemented here");
    assert(abi_callback_is_even(<Opaque>counter, 4) == 1 && abi_callback_is_even(<Opaque>counter, 7) == 0, "a bool result");
    const callback: Callback = counter;
    assert(callback.count() == 10, "and tslang through the C++ interface");
    assert(counter.doubled() == 20, "a method of TypeScript's own");
    assert(abi_callback_pick(<Opaque>counter) == 80025, "C++ calls the overloads implemented here");
    assert(counter.pick(1) == 6 && counter.pick(0.5) == 50, "and tslang picks them too");

    // records by value: in registers, by a pointer to a copy, on the stack, as the ABI passes each
    let c: S12 = [7, 8, 9];
    let e: S24 = [13, 14, 0x987654321];
    assert(methods.byValue(1, [2, 3, 4], [5, 6], c, [0x123456789, 11, 12], e, 16) == 2320580378263, "records passed by value");
    assert(c.a == 7 && e.a == 13, "the callee's copies, not the caller's records");
    assert(methods.pair([0x123456789, 11, 12], 16) == 151488270538, "a 16-byte record by value");
    const flags = methods.flags(5);
    assert(flags.a && !flags.b && flags.c, "a record of three booleans returned");
    const r12 = methods.r12(7);
    assert(r12.a == 7 && r12.b == 14 && r12.c == 21, "a 12-byte record returned");

    // an object on the C++ heap, which C++ deletes: a count written in front of it breaks the delete
    const heap = <Methods>abi_heap_object();
    for (let i = 0; i < 3; i++) {
        assert(copies(heap) == 28, "a C++ object held and let go, uncounted");
    }
    assert(abi_guard_intact(<Opaque>heap) == 1, "and nothing once they are let go");
    abi_delete_object(<Opaque>heap);

    print("done.");
}
