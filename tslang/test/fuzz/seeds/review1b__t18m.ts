class V { constructor(public x: number) {} }
class N { constructor(public c: V) {} }
let n = new N(new V(4));
function child(m: N) { return m.c; }
const c = child(n);
function reset() { n = new N(new V(6)); }
function main() { print(c.x); n.c = new V(5); print(n.c.x); }
