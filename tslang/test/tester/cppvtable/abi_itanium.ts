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

    print("done.");
}
