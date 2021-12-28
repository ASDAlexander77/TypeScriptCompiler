type Person = { age: number; name: string; alive: boolean };
type Age = Person["age"];

type I1 = Person["age" | "name"];

type I2 = Person[keyof Person];

function main() {

    let a: Age = 10.0;

    let b: I1;

    let c: I2;

    print("done.");
}
