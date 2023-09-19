namespace s {
    function f1() {
        function clear() {
            print("clear 1");
            return 1;
        }

        return clear();
    }

    function f2() {
        function clear() {
            print("clear 2");
            return 2;
        }

        return clear();
    }
}

function main() {
    assert(s.f1() == 1);
    assert(s.f2() == 2);
    print("done.")
}
