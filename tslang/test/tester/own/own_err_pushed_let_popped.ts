// -mm=own rejects: `pipe`, a local pushed where it is not null, is the array's element after the
// push (TSLANG 79); popped before it is returned, it would be a freed element.
@final
class Pipe {
    constructor(public id: int) {}
}

@final
class Cache {
    private pipes: Pipe[] = [];

    get(ok: boolean): Pipe | null {
        let pipe: Pipe | null = null;
        if (ok) {
            pipe = new Pipe(1);
        }
        if (pipe) {
            this.pipes.push(pipe);
            this.pipes.pop();
        }
        return pipe;
    }
}

function main() {
    const c = new Cache();
    print(c.get(true) != null);
}
