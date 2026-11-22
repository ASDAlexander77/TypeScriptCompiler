class E {
    readonly f: number = 1;
    reset() { this.f = 0; }
}
function main() { const e = new E(); e.reset(); print(e.f); }
