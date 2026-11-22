class E {
    readonly f: number = 1;
    constructor() { const g = () => { this.f = 2; }; g(); }
}
function main() { print(new E().f); }
