let glb1 = 0;
let x = 0;

function pause(ms: number): void { }

namespace exceptions {
    function immediate(k: number) {
        try {
            pause(1)
            if (k > 0)
                throw k
            pause(1)
            glb1++
        } catch (e: number) {
            assert(e == k)
            glb1 += 10
            if (k >= 10)
                throw e
        } finally {
            x += glb1
        }
    }

    function throwVal(n: number) {
        pause(1)
        if (n > 0)
            throw n
        pause(1)
    }


    function higherorder(k: number) {
        try {
            //[1].map(() => throwVal(k))
            (() => throwVal(k))();
            glb1++
        } catch (e: number) {
            assert(e == k)
            glb1 += 10
            if (k >= 10)
                throw e
        } finally {
            x += glb1
        }
    }

    function lambda(k: number) {
        function inner() {
            try {
                throwVal(k)
                glb1++
            } catch (e: number) {
                assert(e == k)
                glb1 += 10
                if (k >= 10)
                    throw e
            } finally {
                x += glb1
            }
        }
        inner()
    }

    function callingThrowVal(k: number) {
        try {
            pause(1)
            throwVal(k)
            pause(1)
            glb1++
        } catch (e: number) {
            assert(e == k)
            glb1 += 10
            if (k >= 10)
                throw e
        } finally {
            x += glb1
        }
    }

    function nested() {
        try {
            try {
                callingThrowVal(10)
            } catch (e: number) {
                assert(glb1 == 10 && x == 10)
                glb1++
                throw e
            }
        } catch (ee: number) {
            assert(glb1 == 11)
        }
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

    function test4(fn: () => void) {
        try {
            fn()
            return 10
        } catch {
            return 20
        }
    }

    function test5() {
        let n = 0
        for (let k of [0, 1, 2]) {
            try {
                n++
                try {
                    if (k == 1)
                        break
                } catch {
                    n += 1000
                }
            } catch {
                n += 100
            }
        }
        assert(n == 2)
    }

    export function run() {
        print("test exn")
        glb1 = x = 0
        callingThrowVal(1)
        assert(glb1 == 10 && x == 10)
        callingThrowVal(0)
        assert(glb1 == 11 && x == 21)
        callingThrowVal(3)
        assert(glb1 == 21 && x == 42)

        test3(callingThrowVal)
        test3(immediate)
        test3(higherorder)
        test3(lambda)

        glb1 = x = 0
        nested()
        assert(glb1 == 11)

        assert(test4(() => { }) == 10)
        assert(test4(() => { throw "foo" }) == 20)

        test5()

        print("test exn done")
        print("done.");
    }
}

function main() {
    exceptions.run();
}