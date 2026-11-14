// TSLANG_ISSUES 66: a `let` borrowing a field, given a fresh value on some paths only, is still the
// field's on the others: the field overwritten frees it.
function rot(v: number[], a: number) {
    return [v[0] * a, v[1], v[2]];
}
class Cam {
    dir: number[] = [1.0, 0.0, 1.0];
    turn(c: boolean) {
        let dir = this.dir;
        if (c) {
            dir = rot(dir, 2.0);
        }
        this.dir = [5.0, 0.0, 0.0];
        return dir[0];
    }
}
function main() {
    const c = new Cam();
    print(c.turn(false));
    print("done.");
}
