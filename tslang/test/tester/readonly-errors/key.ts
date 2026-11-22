class N { constructor(readonly v: number) {} }
function main() { const n = new N(1); const k: "v" = "v"; n[k] = 7; print(n.v); }
