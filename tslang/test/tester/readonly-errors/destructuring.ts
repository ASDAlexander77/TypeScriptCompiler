class N { constructor(readonly v: number) {} }
function main() { const n = new N(1); [n.v] = [5]; print(n.v); }
