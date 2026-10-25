// @cppname: a C++ declaration, bound by the symbol each target's C++ compiler gives it - one class
// for MSVC (Windows) and Itanium (Linux, Android, MinGW), where @linkname needs each one spelled out.
// `int64_t` is `long` on 64-bit Linux and `long long` on Windows, and mangles as each.
declare function cppname_object(): Opaque;

@cpp
class Methods {
    @cppname("int scaled(int)")
    scaled(x: s32): s32;
    @cppname("int scaled(double)")
    scaled(x: f64): s32;
    @cppname("int get() const")
    get(): s32;
    @cppname("int64_t total(int64_t value)")
    total(x: s64): s64;
}

@cppname("int cppname::sum(int, int)")
declare function sum(a: s32, b: s32): s32;

function main() {
    const methods = <Methods>cppname_object();
    print(methods.scaled(1), methods.scaled(2.5), methods.get(), methods.total(3), sum(5, 6));
}
