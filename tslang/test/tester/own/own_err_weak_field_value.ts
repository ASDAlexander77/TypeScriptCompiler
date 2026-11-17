@value class Point { readonly x: number = 1; }
class Holder { @weak p: Point | null = null; }
function main() { const h = new Holder(); print(h.p === null); }
