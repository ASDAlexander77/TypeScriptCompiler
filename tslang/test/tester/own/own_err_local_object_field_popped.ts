// -mm=own rejects: `a`, pushed on a field of the local object `frame`, is read back from it after the
// push (TSLANG 78); popped first, it would be read from a freed element.
@final
class Alloc {
    constructor(public id: int) {}
}

@final
class Frame {
    allocs: Alloc[] = [];
}

function build() {
    const frame = new Frame();
    const a = new Alloc(1);
    frame.allocs.push(a);
    frame.allocs.pop();
    print(a.id);
}

function main() {
    build();
}
