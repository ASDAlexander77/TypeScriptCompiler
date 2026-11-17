class A { a = 1; }
class B { b = 2; }
class Holder { @weak x: A | B | null = null; }
function main() { const h = new Holder(); print(h.x === null); }
