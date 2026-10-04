// A module with top-level statements, imported (#447): they run when the program starts, before
// the importer's main. Generated at the importer's module level, outside any function, they were
// invalid LLVM IR ("Global is referenced by parentless instruction").
let started = 0;

export function twice(n: number) {
    return n * 2;
}

export function startCount() {
    return started;
}

started = started + 1;
print("module init");
