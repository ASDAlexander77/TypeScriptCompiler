// -mm=own, phase 0 rejects: a declared function's result is not known to be fresh, so holding
// it would be a second owner of something the callee may still own. Compiled, never linked.
declare function makeNumbers(): number[];
function main() {
    const b = makeNumbers();
    print(b.length);
    print("done.");
}
