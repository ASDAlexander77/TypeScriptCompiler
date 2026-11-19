class E { x = 1; }
function f(o: object) { return o instanceof E; }
function main() { print(f(new E())); }
