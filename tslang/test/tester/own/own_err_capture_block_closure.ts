class V { constructor(public x: number) {} get(): number { return this.x; } }
class N { constructor(public v: V) {} }
function f2(): number {
    const n4 = new N(new V(1));
    { let c5 = (): number => n4.v.get(); } return n4.v.x;
}
function main() { print(f2()); }
