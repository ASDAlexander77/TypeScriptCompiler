// -mm=own rejects: a `public` method of a class no other module can name, that overrides a method of
// an exported class. Another module gets a `Base` from `mk()` and calls `keep` through `Base`'s
// vtable, without the method's facts, so the method may not keep its argument (spec 31).
export class V { constructor(public x: number) {} }
export abstract class Base { abstract keep(v: V): void; }
class Der extends Base { saved: V[] = []; public keep(v: V) { this.saved.push(v); } public first() { return this.saved[0].x; } }
let d = new Der();
export function mk(): Base { return d; }
export function first() { return d.first(); }
