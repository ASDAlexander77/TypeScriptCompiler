// @strict-null false
function foo3(results: number[] | undefined) {
    results ||= [];
    if (results) print("has values");
    let a = <number[]>results;
    a.push(1);
}

function foo4(results: number[]) {
    results ??= [];
    if (results) print("has values");
    results.push(1);
}

function main()
{
	foo3(undefined);
	foo4(null);
	print("done.");
}