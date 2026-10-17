// A C variadic function called through a value of it - a `let`, a `typeof printf` parameter -
// is a variadic call, as a call by name is: a double after the format reaches printf (on Win64
// it printed 0.000000 when the call was lowered as a fixed-arity one).
@varargs
declare function printf(format: string): s32;

function viaParam(f: typeof printf) {
    f("param %f\n", 1.5);
    f("mixed %d %f %d %f\n", 7, 2.5, 9, 0.25);
}

function main() {
    printf("direct %f\n", 1.5);

    let p = printf;
    p("let %f\n", 1.5);
    p("no extra arguments\n");

    viaParam(printf);
    viaParam(p);

    // C's default argument promotions: an f32 goes to `...` as a double, an s8 and a boolean
    // as an int
    let x: f32 = 1.5;
    let small: s8 = -3;
    let flag = true;
    printf("promoted %f %d %d\n", x, small, flag);
    p("promoted %f %d %d\n", x, small, flag);
}
