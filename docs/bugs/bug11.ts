namespace LambdaProperty {

    interface IFoo {
        y: number;
        z: number;
        bar: () => number;
        baz: (i: number) => number;
    }

    export function test() {

        let x: IFoo = {
            y: <number>3, z: <number>4, bar: () => {
                return <number>0
            }, baz: (i: number) => i + 1
        }

        // TODO: should i allow to change it?
        /*
        x.bar = () => {
            return x.y
        }
        */

        //assert(x.bar() == 3);
	print(x.baz(42));
        //assert(x.baz(42) == 43);
        x = null // release memory
    }
}

function main() {
    LambdaProperty.test()
    print("done.");
}
