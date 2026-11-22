class C {
    readonly a: number = 0;
    m() { (() => { this.a = 1; })(); }
}
function main() { const c = new C(); c.m(); print(c.a); }
