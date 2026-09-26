// Its reference is relative to this file, as in TypeScript: nested/inner.d.ts. It used to be
// looked for beside the program only, and not found.
/// <reference path="inner.d.ts" />
type OuterT = [o: s32];
