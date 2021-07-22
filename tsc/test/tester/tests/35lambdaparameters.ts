function testLambdasWithMoreParams() {
    function a(f: (x: number, v: string, y: number) => void) {
        f(1, "a" + "X12b", 7);
    }
    a((x: number, v: string, y: number) => {});
}

namespace Arcade1617 {
    class Foo {
        public handlerxx: (win?: boolean) => void;
        run() {
            this.handlerxx();
        }
    }

    function end(win?: boolean) {
        assert(win === undefined, "lp1");
    }

    function test() {
        const f = new Foo();
        f.handlerxx = end;
        f.run();
    }
}

function main() {
    testLambdasWithMoreParams();
    Arcade1617.test();
    print("done.");
}
