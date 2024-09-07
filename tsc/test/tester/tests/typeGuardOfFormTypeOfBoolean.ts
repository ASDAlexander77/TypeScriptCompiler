class C { private p: string };

function main() {

    let str: string = "";
    let bool: boolean = true;
    let num: number = 1;
    let strOrNum: string | number = 1;
    let strOrBool: string | boolean = "";
    let numOrBool: number | boolean = 1
    let strOrNumOrBool: string | number | boolean = "";
    let strOrC: string | C = "";
    let numOrC: number | C = 1;
    let boolOrC: boolean | C = true;
    let c: C = new C();

    //	A type guard of the form typeof x === s, 
    //  where s is a string literal with the value 'string', 'number', or 'boolean',
    //  - when true, narrows the type of x to the given primitive type, or
    //  - when false, removes the primitive type from the type of x.
    if (typeof strOrBool === "boolean") {
        bool = strOrBool; // boolean
    }
    else {
        str = strOrBool; // string
    }
    if (typeof numOrBool === "boolean") {
        bool = numOrBool; // boolean
    }
    else {
        num = numOrBool; // number
    }
    if (typeof strOrNumOrBool === "boolean") {
        bool = strOrNumOrBool; // boolean
    }
    else {
        strOrNum = strOrNumOrBool; // string | number
    }
    if (typeof boolOrC === "boolean") {
        bool = boolOrC; // boolean
    }
    else {
        c = boolOrC; // C
    }

    if (typeof strOrNum === "boolean") {
        let z1: {} = strOrNum; // {}
    }
    else {
        let z2: string | number = strOrNum; // string | number
    }

    // A type guard of the form typeof x !== s, where s is a string literal,
    //  - when true, narrows the type of x by typeof x === s when false, or
    //  - when false, narrows the type of x by typeof x === s when true.
    if (typeof strOrBool !== "boolean") {
        str = strOrBool; // string
    }
    else {
        bool = strOrBool; // boolean
    }
    if (typeof numOrBool !== "boolean") {
        num = numOrBool; // number
    }
    else {
        bool = numOrBool; // boolean
    }
    if (typeof strOrNumOrBool !== "boolean") {
        strOrNum = strOrNumOrBool; // string | number
    }
    else {
        bool = strOrNumOrBool; // boolean
    }
    if (typeof boolOrC !== "boolean") {
        c = boolOrC; // C
    }
    else {
        bool = boolOrC; // boolean
    }

    if (typeof strOrNum !== "boolean") {
        let z1: string | number = strOrNum; // string | number
    }
    else {
        let z2: {} = strOrNum; // {}
    }

    print("done.");

}