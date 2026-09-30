// Run by jit-throw-without-runtime.cmake under the JIT with no --shared-libs: every throw needs
// type_info's vtable, which only TypeScriptRuntime.dll exported, and the JIT loads that for gc alone.
class Failure {
    constructor(public code: number) {}
}

function throwNumber() {
    throw 1;
}

function throwString() {
    throw "text";
}

function throwClass() {
    throw new Failure(3);
}

function main() {
    let caught = 0;

    try { throwNumber(); } catch (e) { caught++; }
    try { throwString(); } catch (e) { caught++; }
    try { throwClass(); } catch (e) { caught++; }

    assert(caught == 3, "every throw is caught");
    print("ALL DONE");
}
