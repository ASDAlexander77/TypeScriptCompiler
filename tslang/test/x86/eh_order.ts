// Unwinding order under exceptions. Prints a trace and never reads a catch variable's value.
// check-x86-eh.sh builds this for 32-bit Windows and compares its output exactly.
function thrower(depth: number) {
    try {
        if (depth == 0) {
            print("throw");
            throw 1;
        }
        thrower(depth - 1);
    } finally {
        print("finally " + depth);
    }
}

function nested() {
    try {
        try {
            throw 1;
        } catch {
            print("inner catch");
            throw 2;   // throw from a catch
        }
    } catch {
        print("outer catch");
    }
}

function rethrow() {
    try {
        try {
            throw 1;
        } catch (e: TypeOf<1>) {
            // Adapted: the plan had `catch (e)`. An untyped catch variable gets a `void *`
            // handler (catchpad ??_R0PEAX@8), which does not match the thrown int (??_R0H@8),
            // so the exception went unhandled and the process ended with exit 127 and no output,
            // at x64 too and with the pre-Phase-3 compiler. That is a catch-variable typing
            // issue, not x86 EH; the typed catch is the idiom 00throw_in_catch.ts uses.
            print("rethrow");
            throw e;
        }
    } catch {
        print("caught rethrown");
    }
}

try {
    thrower(2);
} catch {
    print("caught");
}
nested();
rethrow();
print("done");
