type Uint8Array = u8[];
type Uint16Array = u16[];

function main() {
    const arr = new Uint8Array(10);
    const elem = arr[0];
    print(typeof elem);
    //print(elem);

    let arr2 = new Uint16Array();
    arr2.push(2);
    const elem2 = arr2[0];
    print(typeof elem2);
    //print(elem2);

    print("done.");
}
