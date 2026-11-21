// An object without an optional method of an interface is an error at the object ("failed statement").
interface D { a? (): number; b? (): number; }
function f(d: D) { return 1; }
function main() {
    f({ a() { return 5; } });
}
