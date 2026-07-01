// Example source in TypeScript language.
// `tslang` compiler turns this into mycode.obj, which CMake links
// with main.cpp. Replace with real TypeScript syntax; the symbols exported
// must match the extern "C" declarations in main.cpp.

import './adder'

export function foo_add(a: int, b: int): int {
    const adder = new Adder(a, b);
    return adder.result;
}

export function foo_hello() {
    console.log("hello from foo");
}
