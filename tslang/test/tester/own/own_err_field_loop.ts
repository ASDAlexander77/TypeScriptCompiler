// -mm=own, phase 0 rejects: a field store inside a loop of a string made outside it.
class N { v: number[] = []; s: string = ""; }
function main() {
    let k: number = 3;
    const n = new N();
    const s = "a" + k;
    for (let i = 0; i < 3; i++) {
        n.s = s;
        const filler = "zzzzzzzz" + i;
        print(n.s, filler);
    }
    print("done.");
}
