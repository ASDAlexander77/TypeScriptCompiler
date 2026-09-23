// A while condition that produces no value is an error ("the condition has no value"), not a
// statement generated without its test - see conditionHasValue in MLIRGenImpl.h.

function nothing(): void {}

function main() {
    while (nothing()) {
        print("body");
    }
}
