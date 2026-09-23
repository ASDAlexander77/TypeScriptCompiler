// An integer array literal keeps its `s32` elements, and an array held in a variable is converted
// element by element when it is read as `number[]`. The conversion used to trace the variable back
// to the literal it was declared with and build the array from that literal, so everything done to
// the array since - push, element writes, a new array assigned, a sort - was silently lost.
const moduleArray = [7, 8, 9];
moduleArray[0] = 70;
moduleArray.push(10);
const moduleWidened: number[] = moduleArray;

function widen(a: s32[]): number[] {
    return a;
}

function main() {
    let pushed = [1, 2];
    pushed.push(3);
    pushed[0] = 9;
    const fromLet: number[] = pushed;
    assert(fromLet.length == 3, "let: pushed element counted");
    assert(fromLet[0] == 9, "let: written element");
    assert(fromLet[2] / 2 == 1.5, "let: pushed element, number arithmetic");

    const written = [3, 1, 2];
    written[0] = 42;
    const fromConst: number[] = written;
    assert(fromConst[0] == 42, "const: written element");
    assert(fromConst[1] == 1, "const: untouched element");

    let reassigned = [1, 2];
    reassigned = [5, 6, 7];
    const fromReassigned: number[] = reassigned;
    assert(fromReassigned.length == 3, "reassigned: length");
    assert(fromReassigned[0] == 5, "reassigned: element");

    // no literal to trace back to at all: a parameter
    const fromParameter = widen(pushed);
    assert(fromParameter.length == 3, "parameter: length");
    assert(fromParameter[0] / 2 == 4.5, "parameter: element, number arithmetic");

    const nested = [[1, 2], [3]];
    const fromNested: number[][] = nested;
    assert(fromNested.length == 2, "nested: outer length");
    assert(fromNested[1].length == 1, "nested: inner length");
    assert(fromNested[0][1] / 4 == 0.5, "nested: element, number arithmetic");

    const toUnion: (number | string)[] = written;
    assert(toUnion[0] == 42, "union element type");

    const toAny: any[] = written;
    assert(toAny.length == 3, "any element type: length");

    const empty: s32[] = [];
    const fromEmpty: number[] = empty;
    assert(fromEmpty.length == 0, "empty array");

    // the widened array is a copy: a write to it does not reach the original
    const copy: number[] = written;
    copy[1] = 100;
    assert(written[1] == 1, "copy: original unchanged");
    assert(copy[1] == 100, "copy: written");

    // module level: widened in the global initializer, after the writes before it
    assert(moduleWidened.length == 4, "module: length");
    assert(moduleWidened[0] == 70, "module: written element");
    assert(moduleWidened[3] / 4 == 2.5, "module: pushed element, number arithmetic");

    // module level: widened here, after a write made in this function
    moduleArray[1] = 80;
    const moduleLater: number[] = moduleArray;
    assert(moduleLater[1] == 80, "module: written in function");

    print("done.");
}
