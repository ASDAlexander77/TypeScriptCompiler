function main() {

    let lambda1 = (y = "hello") => { }
    let lambda2 = (x: number, y = "hello") => { }
    let lambda3 = (x: number, y = "hello", ...rest) => { }
    let lambda4 = (y = "hello", ...rest) => { }

    let x = function (str = "hello", ...rest) { }
    let y = (function (num = 10, boo = false, ...rest: any[]) { })()
    let z = (function (num: number, boo = false, ...rest: any[]) { })(10)

    print("done.");
}