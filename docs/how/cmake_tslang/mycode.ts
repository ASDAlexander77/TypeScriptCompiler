// Example source in TypeScript language.
// `tslang` compiler turns this into mycode.obj, which CMake links
// with main.cpp. Replace with real TypeScript syntax; the symbols exported
// must match the extern "C" declarations in main.cpp.

export function foo_add(a: int, b: int): int {
    return a + b;
}

export function foo_hello() {
    print("hello from foo");
}
