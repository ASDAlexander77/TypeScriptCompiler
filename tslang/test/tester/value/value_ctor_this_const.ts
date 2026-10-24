// `this` of a @value constructor is the slot `new` is filling, not a value yet: a copy taken here
// would miss the fields stored after it (`self.a` read 0, not 1).
@value
class V {
    readonly a: number;
    readonly b: number;

    constructor() {
        const self = this;
        this.a = 1;
        this.b = self.a + 1;
    }
}

function main() {
    print(new V().b);
}
