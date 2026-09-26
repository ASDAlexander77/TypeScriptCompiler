// Referenced from several places at once. Before a file reached twice was loaded once, each of
// these declarations was generated again for every path to it, and the module verifier rejected
// the second `COMMON_K` ("redefinition of symbol named 'COMMON_K'").
declare function common_fn(x: s32): s32;
type CommonT = [a: s32, b: s32];
interface CommonI { v: number }
declare const COMMON_K: number;
