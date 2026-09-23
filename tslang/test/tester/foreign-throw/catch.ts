// An untyped catch of a C++ exception tslang did not throw. A pointer exception's type_info is a
// __pointer_type_info, as a thrown tslang class's is, and the catch called the box thunk field
// that only tslang's has - past the end of libstdc++'s, so a `throw "boom"` from C++ made a
// wild call. Only a type_info flagged as tslang's is asked for its thunk now.
declare function foreign_throw_cstr(): void;
declare function foreign_throw_ptr(): void;
declare function foreign_throw_int(): void;

class Own {
    constructor(public n: number) {}
}

function main() {
    let caught = 0;

    try {
        foreign_throw_cstr();
    } catch (e) {
        caught++;
    }

    try {
        foreign_throw_ptr();
    } catch (e) {
        caught++;
    }

    try {
        foreign_throw_int();
    } catch (e) {
        caught++;
    }

    // a tslang class still goes through its box thunk
    try {
        throw new Own(5);
    } catch (e) {
        caught++;
        assert(e instanceof Own, "own class boxed as itself");
    }

    assert(caught == 4, "every exception caught");
    print("done.");
}
