// left and right both reference common: common is loaded once, before either of them.
/// <reference path="left.d.ts" />
/// <reference path="right.d.ts" />

const t: CommonT = [1, 2];
const i: CommonI = { v: 3 };
assert(t[0] + t[1] == i.v);

print("done.");
