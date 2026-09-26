// The program and the module it imports both reference common.d.ts: the import generates the
// module's references into the same module the program's are in, so common was there twice.
/// <reference path="common.d.ts" />
import './module_left'

const t: CommonT = [1, 2];
assert(module_left_fn() + t[0] == 11);

print("done.");
