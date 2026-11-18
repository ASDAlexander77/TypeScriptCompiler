type TypeA = { kind: "a" };
type TypeB = { kind: "b" };
function isTypeA(_value: TypeA | TypeB): value is TypeA {
    return true;
}
function test(input: TypeA | TypeB) {
    if (isTypeA(input)) {
        input;
    }
}
