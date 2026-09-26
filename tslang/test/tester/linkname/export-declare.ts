namespace C {
    @linkname("strlen")
    export declare function len(s: string): index;
}

function main() {
    print(C.len("hello"));
}
