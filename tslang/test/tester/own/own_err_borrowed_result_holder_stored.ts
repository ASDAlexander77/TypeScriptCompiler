// -mm=own rejects: `texture` borrows the result of `Chain.texture`, a place not known, and the store
// into `this.chain` releases a Chain, whose type holds textures: it may free the one borrowed
// (TSLANG 84 refines by type only where the released type cannot hold the borrowed one).
@final
class Tex {
    constructor(public id: int) {}
}

@final
class Chain {
    private textures: Tex[] = [new Tex(1), new Tex(2)];

    @lends("this")
    texture(i: int): Tex {
        return this.textures[i];
    }
}

@final
class App {
    private chain = new Chain();

    frame(): int {
        const texture = this.chain.texture(1);
        this.chain = new Chain();
        return texture.id;
    }
}

function main() {
    print(new App().frame());
}
