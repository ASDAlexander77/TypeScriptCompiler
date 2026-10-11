// The methods of a C++ object made by g++ or clang on x86-64 (abi_host.cpp): two destructor slots;
// a record with a destructor returned through a pointer before `this` (@cppnontrivial), trivial
// records of one or two 8-byte fields in RAX:RDX.

type Ptr = [p: Opaque | null];
type Obj = [p: Opaque | null];
type Two = [a: u64, b: Opaque | null];

declare function abi_object(): Opaque;
declare function abi_is_storage(p: Opaque | null, index: s32): s32;

@cppvtable
class Methods {
    value: s32;
    completeDestructor(): void;             // slot 0
    deletingDestructor(): void;             // slot 1
    @cppnontrivial
    make(x: s32): Ptr;                      // slot 2
    obj(x: s32): Obj;                       // slot 3
    two(x: s32): Two;                       // slot 4
    plain(a: s32, b: s32): s32;             // slot 5
    narrow(b: boolean, u: u8, s: s8): s32;  // slot 6
}

// A C++ interface (abi_host.cpp's Callback) implemented here: C++ calls Counter's methods through the
// vtable tslang makes for it
@cppvtable
class Callback {
    completeDestructor(): void;                 // slot 0
    deletingDestructor(): void;                 // slot 1
    message(severity: u8, text: string): void;  // slot 2
    count(): s32;                               // slot 3
    isEven(x: s32): boolean;                    // slot 4
}

@cppvtable
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
}

declare function abi_call_callback(callback: Opaque): s32;
declare function abi_callback_is_even(callback: Opaque, x: s32): s32;

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

    const counter = new Counter(5);
    assert(abi_call_callback(<Opaque>counter) == 10 && counter.unexpected == 0, "C++ calls the methods implemented here");
    assert(abi_callback_is_even(<Opaque>counter, 4) == 1 && abi_callback_is_even(<Opaque>counter, 7) == 0, "a bool result");
    const callback: Callback = counter;
    assert(callback.count() == 10, "and tslang through the C++ interface");

    print("done.");
}
