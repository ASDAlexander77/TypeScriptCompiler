function main() {

    // The resulting type an array literal expression is determined as follows:
    //     - If the array literal contains no spread elements and is an array assignment pattern in a destructuring assignment (section 4.17.1),
    //       the resulting type is a tuple type constructed from the types of the element expressions.

    let [b1, b2]: [number, number] = [1, 2];

    // The resulting type an array literal expression is determined as follows:
    //      - the resulting type is an array type with an element type that is the union of the types of the
    //        non - spread element expressions and the numeric index signature types of the spread element expressions
    let temp2: [number[], string[]] = [[1, 2, 3], ["hello", "string"]];

    interface tup {
        0: number[] | string[];
        1: number[] | string[];
    }

    print("done.");
}