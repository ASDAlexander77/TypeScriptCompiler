// TSLANG 77: an array moved into a constructor that keeps it, the object pushed, then filled through
// the local: each push fills the object's array (`sets` is that array, by reference).
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
    build(n: int): int {
        for (let i = 0; i < 2; i++) {
            const sets: DSet[] = [];
            this.frames.push(new Frame(i, sets));
            for (let t = 0; t < n; t++) {
                const set = new DSet(t);
                sets.push(set);
            }
        }
        return this.frames[1].sets.length;
    }
}
function main() { const a = new App(); assert(a.build(3) == 3, "sets"); print("done."); }
