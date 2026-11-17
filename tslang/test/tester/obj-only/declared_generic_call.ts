// A generic `declare function` called with an arrow: its specialization is a declaration, which must stay
// external (was given linkonce_odr/private: "'llvm.func' op external functions must have 'external' or
// 'extern_weak' linkage", TSLANG 104). Compiled to an object only: nothing defines `f`.
declare function f<T>(p: (t: T) => T): T;
function main() { f((n: number) => n); }
