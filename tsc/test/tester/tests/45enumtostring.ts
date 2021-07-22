function getABoolean() {
    return !!true;
}

enum SomeEnum {
    One = 1,
    Two = 2,
}

function main() {
    print("test enum to string");

    let enumTest = getABoolean() ? SomeEnum.One : SomeEnum.Two;

    assert(`${enumTest}` === "1", "enum tostring in template");
    assert(enumTest + "" === "1", "enum tostring in concatenation");

    print("done.");
}
