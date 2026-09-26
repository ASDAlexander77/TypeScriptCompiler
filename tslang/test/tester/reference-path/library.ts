// Definitions, not only declarations - what a tsbindgen binding holds. Referenced by program.ts
// directly and through library_user.ts.
const LIB_K = 42;

function lib_fn(x: number) { return x + 1; }

class LibC { v = 1; }

enum LibE { A = 1, B }
