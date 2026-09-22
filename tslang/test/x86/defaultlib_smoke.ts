// The x86 default library (spec Phase 4, As built): a program that leans on the default library itself -
// Array, Map, string and Date methods, plus RegExp, including the replaceAll(regex, ...) overload
// whose native call passed only 3 of the 4 arguments declare function regexp_replace takes.
// tslang does not check declare-function call arity (TypeScript's TS2554), so the undefined 4th
// argument read whatever register or stack slot happened to hold it: register luck at x64, a
// garbage stack slot - and a segfault - at i686. Fixed in the default library, not here (see the
// spec's Phase 4 "As built" note). assert() is a compiler intrinsic (TypeScript_AssertOp /
// AssertOpLowering), not a default-library function - it works fine under --no-default-lib too.
// What this file needs the default library FOR is Array.sort, Map, Date and String.replaceAll,
// which live in the DefaultLib repo's lib.ts, so this file is compiled WITHOUT --no-default-lib.
//
// Kept to patterns the DefaultLib repo's own 150-test suite already exercises: a top-level `let`
// array (a top-level `const` array's .sort() does not mutate it - a separate, pre-existing,
// non-x86 gap) and Map get/size without overwriting an existing key (Map.set on an existing key
// does not update its value - also pre-existing and not x86-specific). Neither is this task's to
// fix; this file only proves the x86 default library runs the paths that already work.
function main() {
    let arr = [5, 3, 8, 1, 9, 2];
    arr.sort((a, b) => a - b);
    let sum = 0;
    for (const n of arr) {
        sum += n;
    }
    assert(sum == 28, "array sum");
    assert(arr[0] == 1 && arr[5] == 9, "array sort");

    const m = new Map<string, string>();
    m.set("a", "alpha");
    m.set("b", "beta");
    assert(m.get("a") == "alpha", "map get");
    assert(m.size == 2, "map size");

    const paragraph = "I think Ruth's dog is cuter than your dog!";
    const byString = paragraph.replaceAll("dog", "monkey");
    assert(byString == "I think Ruth's monkey is cuter than your monkey!", "string replaceAll");

    // Global flag required when calling replaceAll with a regex.
    const regex = /Dog/gi;
    const byRegex = paragraph.replaceAll(regex, "ferret");
    assert(byRegex == "I think Ruth's ferret is cuter than your ferret!", "regex replaceAll");

    const d = new Date(2024, 0, 15);
    assert(d.getFullYear() == 2024, "date year");
    assert(d.getMonth() == 0, "date month");
    assert(d.getDate() == 15, "date day");
    d.setDate(d.getDate() + 20);
    assert(d.getMonth() == 1, "date rollover");

    print("ALL DONE");
}
