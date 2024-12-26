// @strict-null false
class A1 {
    public a: string;
    public b: number;
    public c: boolean;
    public d: any;
    public e: Object;
    public fn(a: string): string {
        return null;
    }
}
class B1 {
    public a: string;
    public b: number;
    public c: boolean;
    public d: any;
    public e: Object;
    public fn(b: string): string {
        return null;
    }
}

class Base {
    private a: string;
    private fn(b: string): string {
        return null;
    }
}
class A2 extends Base { }
class B2 extends Base { }

interface A3 { f(a: number): string; }
interface B3 { f(a: number): string; }

interface A4 { new(a: string): A1; }
interface B4 { new(a: string): B1; }

interface A5 { [x: number]: number; }
interface B5 { [x: number]: number; }

interface A6 { [x: string]: string; }
interface B6 { [x: string]: string; }

function main() {

    let a1: A1;
    let a2: A2;
    let a3: A3;
    let a4: A4;
    let a5: A5;
    let a6: A6;

    let b1: B1;
    let b2: B2;
    let b3: B3;
    let b4: B4;
    let b5: B5;
    let b6: B6;

    let base1: Base;
    let base2: Base;

    // operator <
    let r1a1 = a1 < b1;
    let r1a2 = base1 < base2;
    let r1a3 = a2 < b2;
    let r1a4 = a3 < b3;
    let r1a5 = a4 < b4;
    let r1a6 = a5 < b5;
    let r1a7 = a6 < b6;

    let r1b1 = b1 < a1;
    let r1b2 = base2 < base1;
    let r1b3 = b2 < a2;
    let r1b4 = b3 < a3;
    let r1b5 = b4 < a4;
    let r1b6 = b5 < a5;
    let r1b7 = b6 < a6;

    // operator >
    let r2a1 = a1 > b1;
    let r2a2 = base1 > base2;
    let r2a3 = a2 > b2;
    let r2a4 = a3 > b3;
    let r2a5 = a4 > b4;
    let r2a6 = a5 > b5;
    let r2a7 = a6 > b6;

    let r2b1 = b1 > a1;
    let r2b2 = base2 > base1;
    let r2b3 = b2 > a2;
    let r2b4 = b3 > a3;
    let r2b5 = b4 > a4;
    let r2b6 = b5 > a5;
    let r2b7 = b6 > a6;

    // operator <=
    let r3a1 = a1 <= b1;
    let r3a2 = base1 <= base2;
    let r3a3 = a2 <= b2;
    let r3a4 = a3 <= b3;
    let r3a5 = a4 <= b4;
    let r3a6 = a5 <= b5;
    let r3a7 = a6 <= b6;

    let r3b1 = b1 <= a1;
    let r3b2 = base2 <= base1;
    let r3b3 = b2 <= a2;
    let r3b4 = b3 <= a3;
    let r3b5 = b4 <= a4;
    let r3b6 = b5 <= a5;
    let r3b7 = b6 <= a6;

    // operator >=
    let r4a1 = a1 >= b1;
    let r4a2 = base1 >= base2;
    let r4a3 = a2 >= b2;
    let r4a4 = a3 >= b3;
    let r4a5 = a4 >= b4;
    let r4a6 = a5 >= b5;
    let r4a7 = a6 >= b6;

    let r4b1 = b1 >= a1;
    let r4b2 = base2 >= base1;
    let r4b3 = b2 >= a2;
    let r4b4 = b3 >= a3;
    let r4b5 = b4 >= a4;
    let r4b6 = b5 >= a5;
    let r4b7 = b6 >= a6;

    // operator ==
    let r5a1 = a1 == b1;
    let r5a2 = base1 == base2;
    let r5a3 = a2 == b2;
    let r5a4 = a3 == b3;
    let r5a5 = a4 == b4;
    let r5a6 = a5 == b5;
    let r5a7 = a6 == b6;

    let r5b1 = b1 == a1;
    let r5b2 = base2 == base1;
    let r5b3 = b2 == a2;
    let r5b4 = b3 == a3;
    let r5b5 = b4 == a4;
    let r5b6 = b5 == a5;
    let r5b7 = b6 == a6;

    // operator !=
    let r6a1 = a1 != b1;
    let r6a2 = base1 != base2;
    let r6a3 = a2 != b2;
    let r6a4 = a3 != b3;
    let r6a5 = a4 != b4;
    let r6a6 = a5 != b5;
    let r6a7 = a6 != b6;

    let r6b1 = b1 != a1;
    let r6b2 = base2 != base1;
    let r6b3 = b2 != a2;
    let r6b4 = b3 != a3;
    let r6b5 = b4 != a4;
    let r6b6 = b5 != a5;
    let r6b7 = b6 != a6;

    // operator ===
    let r7a1 = a1 === b1;
    let r7a2 = base1 === base2;
    let r7a3 = a2 === b2;
    let r7a4 = a3 === b3;
    let r7a5 = a4 === b4;
    let r7a6 = a5 === b5;
    let r7a7 = a6 === b6;

    let r7b1 = b1 === a1;
    let r7b2 = base2 === base1;
    let r7b3 = b2 === a2;
    let r7b4 = b3 === a3;
    let r7b5 = b4 === a4;
    let r7b6 = b5 === a5;
    let r7b7 = b6 === a6;

    // operator !==
    let r8a1 = a1 !== b1;
    let r8a2 = base1 !== base2;
    let r8a3 = a2 !== b2;
    let r8a4 = a3 !== b3;
    let r8a5 = a4 !== b4;
    let r8a6 = a5 !== b5;
    let r8a7 = a6 !== b6;

    let r8b1 = b1 !== a1;
    let r8b2 = base2 !== base1;
    let r8b3 = b2 !== a2;
    let r8b4 = b3 !== a3;
    let r8b5 = b4 !== a4;
    let r8b6 = b5 !== a5;
    let r8b7 = b6 !== a6;

    print("done.");
}