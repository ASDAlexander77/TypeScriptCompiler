/// <reference path="nested/outer.d.ts" />

const o: OuterT = [1];
const i: InnerT = [2];
assert(o[0] + i[0] == 3);

print("done.");
