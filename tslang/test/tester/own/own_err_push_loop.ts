// -mm=own, phase 0 rejects: a push inside a loop would give the one instance a new owner each iteration.
class C { x: number = 5; v: number[] = [1, 2]; }
function main() {
    const arr: C[] = [];
    const c = new C();
    for (let i = 0; i < 3; i++) arr.push(c);
    print(arr[0].x, arr[2].x);
    print("done.");
}
