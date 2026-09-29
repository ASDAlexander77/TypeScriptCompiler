// -mm=own, phase 0 rejects: the array and `d` would both own the instance.
class C { x: number = 5; v: number[] = [1, 2]; }
function main() {
    const arr: C[] = [];
    const c = new C();
    arr.push(c);
    let d = c;
    print(arr[0].x, d.x);
    print("done.");
}
