// Regression test for the chained-sibling-method-return-gap (see that memory
// for the full 3-layer mechanism). A method inside an object literal that
// `return`s (or otherwise uses) a sibling method's call result used to fail
// discovery of the ENCLOSING function's own return type/captured vars,
// because a sibling's prototype isn't registered into the literal's storage
// type until its own discovery completes - and a call to an
// not-yet-registered sibling can't produce a value during the speculative
// (dummyRun) discovery pass. Root cause: discovery always tries to INFER a
// return type from the body even when the method already has an EXPLICIT
// one, so a legitimately-unresolvable inferred value ("none") was wrongly
// treated as "discovery failed to converge".
function main() {
    let calc = {
        base: 10,
        scale(factor: number): number {
            return this.base * factor;
        },
        scaleAndAdd(factor: number, extra: number): number {
            return this.scale(factor) + extra;
        }
    };

    assert(calc.scaleAndAdd(2, 3) == 23);

    print("done.");
}
