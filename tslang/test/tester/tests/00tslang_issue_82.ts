// TSLANG 82: a borrow of a call's result, whose place is not known (`this.chain.texture(1)`), across a
// store that releases a value whose type cannot hold it (a @final Pipe, holding no texture): the store
// does not free it.
@final
class Tex {
    constructor(public id: int) {}
}
@final
class Chain {
    private textures: Tex[] = [new Tex(1), new Tex(2)];
    @lends("this")
    texture(i: int): Tex { return this.textures[i]; }
}
@final
class Pipe {
    constructor(public id: int) {}
}
@final
class App {
    private chain = new Chain();
    private pipes: (Pipe | null)[] = [null, null];
    pipeline(k: int): Pipe | null {
        const made = this.pipes[k];
        if (made) {
            return made;
        }
        const p = new Pipe(k);
        this.pipes[k] = p;
        return this.pipes[k];
    }
    frame(): int {
        const texture = this.chain.texture(1);
        const p = this.pipeline(0);
        return texture.id + (p ? p.id : 0);
    }
}
function main() {
    const a = new App();
    assert(a.frame() == 2, "frame");
    print("done.");
}
