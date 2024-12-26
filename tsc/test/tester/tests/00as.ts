// @strict-null false
// TODO: add cast 'string[]' -> 'any[]'
//function tag(...x: any[]): any
function tag(...x: string[]): string
{
    return x[0];
}

function main() {
    // ====== 1

    let x = 0;
    // Must emit as (x + 1) * 3
    assert (((x + 1 as number) * 3) == 3);

    // ====== 2

    let as = 43;
    let x2 = undefined as number;
    // TODO: null access
    //let y = (null as string).length;
    let y = (null as string)?.length;

    // Should parse as a union type, not a bitwise 'or' of (32 as number) and 'string'
    let j = 32 as number | string;
    j = '';

    // ====== 3
    let x3 = 23 as string;
	print(x3, typeof x2);

    // ====== 4
    var a = `${123 + 456 as number}`;
    var b = `leading ${123 + 456 as number}`;
    var c = `${123 + 456 as number} trailing`;
    var d = `Hello ${123} World` as string;
    var e = `Hello` as string;
    var f = 1 + `${1} end of string` as string;
    var g = tag `Hello ${123} World` as string;
    var h = tag `Hello` as string;

    print("done.");
}