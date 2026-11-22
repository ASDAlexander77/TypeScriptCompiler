class N { constructor(readonly v: number) {} }
function main() { const n = new N(1); n.v = 2; print(n.v); }
