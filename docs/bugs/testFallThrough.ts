
function _panic(code: number): void {
    assert(false, `error code: ${code}`);
}

function isClose(x: number, y: number): void {
    if (isNaN(x) && isNaN(y)) return
    const d = Math.abs(x - y)
    if (d < 0.00000001 || d / Math.abs(x + y) < 0.00001) return
    console.log(x, " !== ", y, "!")
    _panic(108)
}

function isEq(x: any, y: any): void {
    // console.log(x, " === ", y, "?")
    if (x !== y) {
        console.log(`fail: ${x} !== ${y}`)
        _panic(109)
    }
}

function strEq(a: string, b: string) {
    if (a !== b) {
        console.log(`fail: '${a}' !== '${b}'`)
        _panic(110)
    }
}

let x = 0
let glb1 = 0

function testFallThrough(x: number) {
    let r = ""
    switch (x) {
        // @ts-ignore
        default:
            r += "q"
        // fallthrough
        case 6:
        // @ts-ignore
        case 7:
            r += "x"
        // fallthrough
        case 8:
            r += "y"
            break
        case 10:
            r += "z"
            break
    }
    return r
}

isEq(testFallThrough(100), "qxy")
isEq(testFallThrough(6), "xy")
isEq(testFallThrough(7), "xy")
isEq(testFallThrough(8), "y")
isEq(testFallThrough(10), "z")

console.log("all OK")
