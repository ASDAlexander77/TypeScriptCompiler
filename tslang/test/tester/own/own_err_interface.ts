// -mm=own, phase 0 rejects: boxing an instance into an interface counts as a second owner beside
// the temporary (interface and `any` boxing are phase 3).
interface I { x: number; }
class C implements I { x: number = 5; v: number[] = [1]; }
function show(i: I) { return i.x; }
function main() {
    print(show(new C()));
    const i: I = new C();
    print(i.x);
    print("done.");
}
