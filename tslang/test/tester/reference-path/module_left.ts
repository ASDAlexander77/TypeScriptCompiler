// A module that references common.d.ts too; see import_and_reference.ts.
/// <reference path="common.d.ts" />

export function module_left_fn() { const t: CommonT = [10, 20]; return t[0]; }
