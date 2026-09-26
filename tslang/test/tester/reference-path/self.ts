// A file referencing itself is already loaded. It used to be loaded again, forever.
/// <reference path="self.ts" />

function self_fn() { return 1; }

assert(self_fn() == 1);

print("done.");
