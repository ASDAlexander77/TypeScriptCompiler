class Val
{
    constructor(public value: number) {        
    }

    [Symbol.toPrimitive](hint: string) {
        if (hint === 'number') {
            return this.value;
        }
        
        return <string>this.value;
    }   
}

const v = new Val(1);
assert((v + 1) == 2);
assert((1 + v) == 2);
assert((v + v) == 2);

print("done.");
