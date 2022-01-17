let glb1 = 0;
let x = 0;

function pause(ms: number): void { }

namespace exceptions {

    function throwVal(n: number) {
			print("throw", n);
        pause(1)
        if (n > 0)
            throw n
        pause(1)
    }

    function lambda(k: number) {
        function inner() {
		print("inner inside", k);
        }
	print("inner");
        inner()
	print("exit inner");
    }

    function test3(fn: (k: number) => void) {
        glb1 = x = 0
        fn(1)
        assert(glb1 == 10 && x == 10)
        fn(0)
        assert(glb1 == 11 && x == 21)
        fn(3)
        assert(glb1 == 21 && x == 42)
    }

    export function run() {
        print("test")

        glb1 = x = 0
        test3(lambda)

        print("done.");
    }
}

function main() {
    exceptions.run();
}