namespace exceptions {

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

        assert(test4(() => { }) == 10)
        assert(test4(() => { throw "foo" }) == 20)

        print("done.");
    }
}

function main() {
    exceptions.run();
}