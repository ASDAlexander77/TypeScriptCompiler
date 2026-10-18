// A C variadic function is not a plain function value: a call through the plain type would
// pass the extra arguments the fixed way.
@varargs
declare function printf(format: string): s32;

function main() {
    let q: (format: string) => s32 = printf;
    q("x\n");
}
