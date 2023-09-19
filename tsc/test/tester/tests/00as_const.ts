function main() {

    const const_obj = {
        val1: "Hello"
    };

    print("val1: ", const_obj.val1);

    let obj = const_obj;

    obj.val1 = "new val 1";

    print("val1: ", obj.val1);

    print("val1 as const: ", (obj as const).val1);

    // TODO: the following code will error
    //(obj as const).val1 = "new val 2";

    print("val1 as const: ", (obj as const).val1);

    print("done.");
}
