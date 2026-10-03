// A module-level variable initialised with a string concatenation of constants, read by a function
// (#469). The initializer looked constant, so it was placed where nothing may run, and the
// concatenation (and the number's conversion to a string) failed to compile there: "ops with side
// effects not allowed in global initializers".
let S = "text" + 4;
const C = "text" + "four";
let U: string | null = "nullable" + 5;
let N = "" + 6.5;
let P = "plain";

function main() {
    assert(S == "text4", "a let from a constant concatenation");
    assert(C == "textfour", "a const from two string constants");
    assert(U == "nullable5", "a nullable string from a constant concatenation");
    assert(N == "6.5", "a number converted to a string");
    assert(P == "plain", "a plain string constant stays as it is");

    S = S + "!";
    assert(S == "text4!");

    print("done.");
}
