// Root-level executable code with NO `function main()` anywhere in the file.
// When there's no explicit `main`, generateGlobalEntryCode (MLIRGenModule.cpp)
// gathers every statement that isn't inside a function/class body and compiles
// it AS the synthesized `main` function itself - so this exercises the same
// control-flow/closure/exception codegen every other test exercises inside an
// explicit main(), just routed through the auto-synthesis path instead.
// Also: a `let`/`const` declared directly at this level is module-scope
// storage (like a static field), not a stack local - so the closure test
// below captures a GLOBAL, not a stack frame slot.

let sum = 0;
for (let i = 0; i < 5; i++) {
    if (i == 2) continue;
    if (i == 4) break;
    sum = sum + i;
}
assert(sum == 4, "for-if-continue-break");

let count = 0;
while (count < 3) {
    count = count + 1;
}
assert(count == 3, "while");

let label = "";
switch (count) {
    case 1:
        label = "one";
        break;
    case 3:
        label = "three";
        break;
    default:
        label = "other";
        break;
}
assert(label == "three", "switch");

let caught = false;
try {
    throw 42.0;
} catch (v: number) {
    caught = v == 42;
} finally {
    sum = sum + 100;
}
assert(caught, "try-catch");
assert(sum == 104, "finally-ran");

let counter = 0;
const increment = () => {
    counter = counter + 1;
};
increment();
increment();
assert(counter == 2, "closure-capture-root-let");

class Counter {
    value: number = 0;
    increment(): void {
        this.value = this.value + 1;
    }
}

const c = new Counter();
c.increment();
c.increment();
c.increment();
assert(c.value == 3, "class-at-root");

print("done.");
