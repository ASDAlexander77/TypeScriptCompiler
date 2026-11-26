// A static block of a class declared in a function, reading the function's local: the static block runs at the start
// of the program - an error, not "'ts.Store' op using value defined outside the region" (TSLANG 197).
function f() {
    const v = 5;
    class H {
        static x: number;
        static {
            H.x = v;
        }
    }

    return H.x;
}

function main() {
    f();
}
