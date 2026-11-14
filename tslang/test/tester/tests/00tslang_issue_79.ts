// TSLANG 79: a local pushed only where it is not null (`let pipeline = ...; if (pipeline)
// this.pipelines.push(pipeline); return pipeline;`) reads the array's element after the push, and null
// where the test failed: the method returns a borrow of `this` on every way.
@final
class Pipe {
    constructor(public id: int) {}
}
@final
class Cache {
    private keys: int[] = [];
    private pipes: Pipe[] = [];

    get(key: int, ok: boolean): Pipe | null {
        for (let i = 0; i < this.keys.length; i++) {
            if (this.keys[i] == key) {
                return this.pipes[i];
            }
        }
        let pipe: Pipe | null = null;
        if (ok) {
            pipe = new Pipe(key);
        }
        if (pipe) {
            this.keys.push(key);
            this.pipes.push(pipe);
        }
        return pipe;
    }
}
function main() {
    const c = new Cache();
    const a = c.get(3, true);
    const b = c.get(3, true);
    const n = c.get(4, false);
    assert((a ? a.id : -1) == 3 && (b ? b.id : -1) == 3 && n == null, "cache");
    print("done.");
}
