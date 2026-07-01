class Val {
    constructor(public value: number) {
    }

    [Symbol.toPrimitive](hint: string) {
        if (hint === 'number') {
            return this.value;
        }

        return <string>this.value;
    }
}
    
let a2: string | number | Val = new Val(1);
let b2: string | number | Val = '1';

let c2 = a2 + b2;
print(c2);

assert(c2 == "11");

print("done.");