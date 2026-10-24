// ...nor a copy a closure reads later: `self.a` read 0, not 7
@value
class V {
    readonly a: number;
    readonly f: () => number;

    constructor() {
        const self = this;
        this.f = () => self.a;
        this.a = 7;
    }
}

function main() {
    print(new V().f());
}
