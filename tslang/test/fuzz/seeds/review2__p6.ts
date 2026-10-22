interface S { roughness: number; }
let a = new Shared<S>({ roughness: 2 });
let k = a;
class Junk { a = -1; b = -2; }
function churn() { let keep: Junk[] = []; for (let i = 0; i < 1000; i++) { keep.push(new Junk()); } return keep.length; }
function main() {
    a = new Shared<S>({ roughness: 3 });
    churn();
    print(k.value.roughness, a.value.roughness);
    print("done.");
}
