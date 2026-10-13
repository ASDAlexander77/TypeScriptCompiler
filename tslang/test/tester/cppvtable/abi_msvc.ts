// The methods of a C++ object made by MSVC (abi_host.cpp): one destructor slot, and every record
// returned through a pointer after `this`.

type Ptr = [p: Opaque | null];
type Obj = [p: Opaque | null];
type Two = [a: u64, b: Opaque | null];

declare function abi_object(): Opaque;
declare function abi_is_storage(p: Opaque | null, index: s32): s32;

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

    print("done.");
}
