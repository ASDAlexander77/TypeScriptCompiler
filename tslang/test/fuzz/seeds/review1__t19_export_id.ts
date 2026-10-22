class V { constructor(public x: number) {} }
function id(v: V) { return v; }
export const c = id(new V(1));
function main() { print(c.x); }
