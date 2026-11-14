// -mm=own rejects: `sets`, moved into a frame pushed on `this.frames`, is that frame's array; the
// frame popped, the push through `sets` would fill a freed array (TSLANG 77).
@final
class DSet {
    constructor(public id: int) {}
}

@final
class Frame {
    constructor(public view: int, public sets: DSet[]) {}
}

@final
class App {
    frames: Frame[] = [];

    build() {
        const sets: DSet[] = [];
        this.frames.push(new Frame(0, sets));
        this.frames.pop();
        sets.push(new DSet(1));
    }
}

function main() {
    new App().build();
}
