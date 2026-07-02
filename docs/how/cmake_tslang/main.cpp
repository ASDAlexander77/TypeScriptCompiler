// Declare the symbols your .foo object exports.
// Use extern "C" so names match your compiler's output (no C++ mangling).
extern "C" int  foo_add(int a, int b);
extern "C" void foo_hello(void);

#include <cstdio>

int main() {
    foo_hello();
    std::printf("foo_add(2,3) = %d\n", foo_add(2, 3));
    return 0;
}
