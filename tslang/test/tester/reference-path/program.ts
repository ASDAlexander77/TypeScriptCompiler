/// <reference path="library.ts" />
/// <reference path="library_user.ts" />

assert(lib_user_fn() == 43);
assert(LIB_K == 42);
assert(new LibC().v == 1);
assert(LibE.B == 2);

print("done.");
