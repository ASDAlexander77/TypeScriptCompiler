// One file under three names: it is the file that counts, not how the reference spells it.
/// <reference path="common.d.ts" />
/// <reference path="./common.d.ts" />
/// <reference path="../reference-path/common.d.ts" />

const t: CommonT = [6, 7];
assert(t[0] == 6);

print("done.");
