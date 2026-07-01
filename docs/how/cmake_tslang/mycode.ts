// Example source in your custom language.
// Your `fooc` compiler turns this into mycode.obj, which CMake links
// with main.cpp. Replace with real FOO syntax; the symbols exported
// must match the extern "C" declarations in main.cpp.

export function foo_add(a, b) {
    return a + b;
}

export function foo_hello() {
    print("hello from foo");
}
