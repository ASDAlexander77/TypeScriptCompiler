function main() {
    let a: string | undefined
    let b: string | undefined
    let c: string | undefined

    let d: number | undefined
    let e: number | undefined
    let f: string | undefined

    let g: 0 | 1 | 42
    let h: 0 | 1 | 42
    let i: 0 | 1 | 42


    a &&= "foo"
    b ||= "foo"
    c ??= "foo"


    d &&= 42
    e ||= 42
    f ??= "42" // must be ref type

    g &&= 42
    h ||= 42
    i ??= 42

    print("done.");
}