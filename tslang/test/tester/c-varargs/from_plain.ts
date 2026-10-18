// Nor is a plain function a C variadic one: a variadic call would hand it its arguments the C
// `...` way.
@varargs
declare function printf(format: string): s32;

function viaParam(f: typeof printf) {
    f("%d\n", 1);
}

function main() {
    viaParam((format: string) => 0);
}
