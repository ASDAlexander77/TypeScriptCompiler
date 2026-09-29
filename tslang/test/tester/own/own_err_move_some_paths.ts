// -mm=own, phase 1 rejects: `a` moves into `b` on one path only, and the release after the `if`
// is reached from both. Keeping it double-frees on the path that moved; erasing it leaks on the
// other. Splitting it (drop elaboration) is not in this phase.
class C { x: number = 5; }
function main(argc: number) {
    const a = new C();
    if (argc > 5) {
        let b = a;
        print(b.x);
    }
    print("done.");
}
