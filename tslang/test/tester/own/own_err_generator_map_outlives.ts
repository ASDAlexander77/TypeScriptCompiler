// -mm=own, phase 7b rejects: the generator `.map` builds borrows `a`, which is assigned while the
// generator is still used.
function main() {
    let a: number[] = [1, 2, 3];
    const it = a.map((x) => x * 2);
    a = [4.5];
    print(it.next().value);
}
