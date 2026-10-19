// A yield inside an expression: what the expression computed before the yield ("a" + i) would be
// an SSA value of the turn that returned, gone when the generator resumes. It crashed at run time.
function* gen() {
    for (let i = 0; i < 2; i++) {
        const t = ("a" + i) + (yield 1);
        print(t);
    }
}

function main() {
    for (const v of gen()) print(v);
}
