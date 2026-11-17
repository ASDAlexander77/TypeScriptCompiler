class H { constructor(public x: number) {} }
@value
interface One { readonly h: H; readonly n: number; }
function f(c: boolean) { let x = new H(5); let o: One = c ? { h: x, n: 1 } : { h: new H(1), n: 2 }; return o.h.x + x.x; }
function main() { print(f(true)); print("done."); }
