// @strict-null false
// TODO: not finished - testGenRef is not compiled (ignored)
function testGenRef<T>(v: T) {
    let x = v
    // test that clear() also gets generalized
    function clear() {
        x = null
    }
    clear()
}

function testGenRefOuter() {
    print("testGenRefOuter");
    testGenRef(12)
    testGenRef("fXa" + "baa")
}

function main()
{
    testGenRefOuter()
    print("done.")
}
