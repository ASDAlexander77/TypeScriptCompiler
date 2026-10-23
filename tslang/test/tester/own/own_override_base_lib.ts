// The library own_err_override_imported_base and own_err_override_imported_class import: an abstract
// and a concrete base class whose `keep` it calls itself.
export class V { constructor(public x: number) {} }
export abstract class Base { abstract keep(v: V): void; }
export class Concrete { keep(v: V): void { } }
export function run(b: Base) { const a = new V(5); b.keep(a); }
export function runConcrete(b: Concrete) { const a = new V(5); b.keep(a); }
