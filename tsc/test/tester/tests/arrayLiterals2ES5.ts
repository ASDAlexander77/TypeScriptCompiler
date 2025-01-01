function main() {
    // ElementList:  ( Modified )
    //      Elisionopt   AssignmentExpression
    //      Elisionopt   SpreadElement
    //      ElementList, Elisionopt   AssignmentExpression
    //      ElementList, Elisionopt   SpreadElement

    // SpreadElement:
    //      ...   AssignmentExpression

    let a0 = [, , 2, 3, 4]
    let a1 = ["hello", "world"]
    let a2 = [, , , ...a0, "hello"];
    let a3 = [, , ...a0]
    let a4 = [() => 1,];
    let a5 = [...a0, ,]

    // Each element expression in a non-empty array literal is processed as follows:
    //    - If the array literal contains no spread elements, and if the array literal is contextually typed (section 4.19)
    //      by a type T and T has a property with the numeric name N, where N is the index of the element expression in the array literal,
    //      the element expression is contextually typed by the type of that property.

    // The resulting type an array literal expression is determined as follows:
    //     - If the array literal contains no spread elements and is contextually typed by a tuple-like type,
    //       the resulting type is a tuple type constructed from the types of the element expressions.

    let b0: [any, any, any] = [undefined, null, undefined];
    let b1: [number[], string[]] = [[1, 2, 3], ["hello", "string"]];

    // The resulting type an array literal expression is determined as follows:
    //     - If the array literal contains no spread elements and is an array assignment pattern in a destructuring assignment (section 4.17.1),
    //       the resulting type is a tuple type constructed from the types of the element expressions.

    let [c0, c1] = [1, 2];        // tuple type [number, number]
    let [c2, c3] = [1, 2, true];  // tuple type [number, number, boolean]

    // The resulting type an array literal expression is determined as follows:
    //      - the resulting type is an array type with an element type that is the union of the types of the
    //        non - spread element expressions and the numeric index signature types of the spread element expressions
    let temp = ["s", "t", "r"];
    let temp1 = [1, 2, 3];
    let temp2: [number[], string[]] = [[1, 2, 3], ["hello", "string"]];
    let temp3 = [undefined, null, undefined];
    let temp4 = [];

    //TODO: array is type, we can't find it as identifier, do we need to improve it?
    //interface myArray extends Array<Number> { }
    //interface myArray2 extends Array<Number | String> { }
    type Array<T> = T[];
    type myArray = Array<Number>;
    type myArray2 = Array<Number | String>;
    let d0 = [1, true, ...temp,];  // has type (string|number|boolean)[]
    let d1 = [...temp];            // has type string[]
    let d2: number[] = [...temp1];
    let d3: myArray = [...temp1];
    let d4: myArray2 = [...temp, ...temp1];
    let d5 = [...temp3];
    let d6 = [...temp4];
    let d7 = [...[...temp1]];
    let d8: number[][] = [[...temp1]]
    let d9 = [[...temp1], ...["hello"]];

    print("done.");
}