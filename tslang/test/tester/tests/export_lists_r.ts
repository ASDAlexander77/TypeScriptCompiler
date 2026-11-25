// TSLANG 191: a module that re-exports another's declarations (see import_export_lists.ts)
export { add as sum, Point } from "./export_lists_m";
export * from "./export_lists_m";
export * as M from "./export_lists_m";
export { default as theAnswer } from "./export_lists_m";

export function twiceSum(a: number, b: number): number {
    return 2 * (a + b);
}
