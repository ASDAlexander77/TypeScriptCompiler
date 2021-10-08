let glb1 = 0;
let x = 0;

function pause(ms: number): void {}

namespace exceptions {

    function throwVal(n: number) {
	print("throw", n);
        pause(1)
        if (n > 0)
            throw n
        pause(1)
    }

    function callingThrowVal(k: number) {
	print("calling throw");
        try {
            pause(1)
            throwVal(k)
            pause(1)
            glb1++
        } catch (e:number) {
	    print("caught", e, k);
            assert(e == k)
            glb1 += 10
            if (k >= 10) {
		print("rethrow", e, k);
                throw e
	    }
        } finally {
	    print("finally", e, k);
            x += glb1
        }
    }

    function nested() {
	print("nested");
        try {
            try {
                callingThrowVal(10)
            } catch (e:number) {
		print("nested caught", e);
                assert(glb1 == 10 && x == 10)
                glb1++
		print("throw again", e);
                throw e
            }
        } catch (ee:number) {
            assert(glb1 == 11)
        }
    }

    function test4(fn: () => void) {
        try {
            fn()
            return 10
        } catch {
            return 20
        }
    }

    export function run() {
        print("test exn")

        glb1 = x = 0
        nested()
        assert(glb1 == 11)

/*
        assert(test4(() => { }) == 10)
        assert(test4(() => { throw "foo" }) == 20)
*/

        print("test exn done")
    }
}

function main()
{
  exceptions.run();
}