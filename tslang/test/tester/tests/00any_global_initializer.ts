// A module-level variable given a value as `any`, or one read from an `any`, is set when the program
// starts: the box (an allocation) and the checked read were left in the global's constant initializer,
// which LLVM refused ("ops with side effects not allowed in global initializers").
let n: any = 5;
let s: any = "text";
let o: any = { x: 1 };
const back = n as number;
let u: unknown = 7;

function main() {
    assert(n == 5);
    assert(s == "text");
    assert(back == 5);
    assert((u as number) == 7);
    print("done.");
}
