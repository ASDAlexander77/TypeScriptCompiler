class C {
    readonly a: number = 0;
    constructor() { const f = () => { (() => { this.a = 1; })(); }; f(); }
}
function main() { print(new C().a); }
