class Holder { @weak names: string[] = []; }
function main() { const h = new Holder(); print(h.names.length); }
