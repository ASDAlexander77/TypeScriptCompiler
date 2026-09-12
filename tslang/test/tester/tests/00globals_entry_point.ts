// A root made only of declarations and variable statements - no expression statement, no
// user-written main(). That still has to produce a runnable program: the initializers run
// from the global constructors, but without an entry point there is nothing for the JIT to
// call or the CRT to link against, and the run failed with "Symbols not found: [ main ]".
//
// Nothing here may be a *code* statement, or the file stops testing the case: an expression
// statement at the root makes an entry function get built for its own sake. So "done." is
// printed from a constructor, reached through the global-constructor path.
//
// The variables are exported to give them external linkage. Without it nothing reads them,
// and at --opt_level=3 LLVM drops both the globals and their constructors - side effect and
// all - leaving a run with no output that this test could not tell from a broken one.

class Greeter {
    constructor(public what: string) {
        print(what);
    }
}

export const first = new Greeter("hello");

export let second = new Greeter("done.");
