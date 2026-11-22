class N { constructor(readonly v?: number) {} }
function main() { const n = new N(1); delete n.v; print(n.v); }
