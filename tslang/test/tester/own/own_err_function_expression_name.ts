// -mm=own refuses a function expression returning its `this` (callWithSpread in the corpus); what
// this test pins is that the message says "this function expression", not MLIRGen's symbol for it.
interface X {
    foo(x: number): X;
}

function main() {
    let obj: X = {
        foo: function (x: number): X {
            print(x);
            return this;
        }
    };

    obj.foo(1);
}
