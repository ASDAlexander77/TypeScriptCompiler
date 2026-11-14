// TSLANG_ISSUES 66: a `let` borrowing a field, or the fresh value it is given on some paths, cannot be
// stored: on the other paths it is the field's.
function rot(v: number[], a: number) {
    return [v[0] * a, v[1], v[2]];
}
class Cam {
    dir: number[] = [1.0, 0.0, 1.0];
    other: number[] = [];
    turn(c: boolean) {
        let dir = this.dir;
        if (c) {
            dir = rot(dir, 2.0);
        }
        this.other = dir;
    }
}
function main() {
    const c = new Cam();
    c.turn(true);
    print("done.");
}
