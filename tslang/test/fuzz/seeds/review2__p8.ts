class C { roughness = 2; constructor(r: number) { this.roughness = r; } }
let a = new Shared<C>(new C(2));
let k = a;
class Junk { a = -1; b = -2; }
function churn() { let keep: Junk[] = []; for (let i = 0; i < 1000; i++) { keep.push(new Junk()); } return keep.length; }
function main() {
    k = a;
    a = new Shared<C>(new C(3));
    churn();
    print(k.value.roughness, a.value.roughness);
    print("done.");
}
