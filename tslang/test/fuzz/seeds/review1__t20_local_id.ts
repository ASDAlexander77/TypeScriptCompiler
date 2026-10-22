class V { constructor(public x: number) {} }
function id(v: V) { return v; }
const c = id(new V(1));
function main() { print(c.x); }
