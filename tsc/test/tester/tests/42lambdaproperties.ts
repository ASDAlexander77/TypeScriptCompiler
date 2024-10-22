namespace LambdaProperty {

    interface IFoo {
        y: number;
        z: number;
        bar: () => number;
        baz: (i: number) => number;
    }

    export function test() {

        let x: IFoo = {
            y: 3, z: 4, bar: () => {
                return 0.0
            }, baz: (i: number) => i + 1
        }

        // TODO: should i allow to change it?
        /*
        x.bar = () => {
            return x.y
        }
        */

        /*
        x.bar = () => {
            // TODO: finish it, we can't convert BoundMethod to method, x is captured, not this parameter
            return 3.0//x.y
        }
        */

        //assert(x.bar() == 3);
        assert(x.baz(42) == 43);
        x = null // release memory
    }
}

function main() {
    LambdaProperty.test()
    print("done.");
}
