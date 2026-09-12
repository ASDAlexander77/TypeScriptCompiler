// The library half of the entry-point pair. Its root only declares things and initializes a
// variable, which is what a library looks like - nothing here is a program, so nothing here
// needs a `main`. It is compiled with --emit=obj, the same action the program half uses, so
// the emit action cannot tell the two apart: only --entry-point does, and the program half
// gets it.
//
// Once a root variable statement alone was enough to ask for an entry point, this file grew a
// `main` too and the link failed with "duplicate symbol: main". Nothing at the root may be a
// code statement, or an entry function gets built for its own sake and the file stops testing
// this.

export let counter = 41;

export function bump() {
    counter = counter + 1;
    return counter;
}
