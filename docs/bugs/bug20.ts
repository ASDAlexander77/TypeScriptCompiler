function main() {

    let strOrNum: string | number = 1;

    if (typeof strOrNum === "boolean") {
        let z1: {} = strOrNum; // {}
    }
    else {
        let z2: string | number = strOrNum; // string | number
    }

    print("done.");

}