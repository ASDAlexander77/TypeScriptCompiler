let count_created = 0;
let count_disp = 0;
function loggy(id: string) {
    print(`Creating ${id}`);
    count_created++;
    return {
        [Symbol.dispose]() {
            print(`Disposing ${id}`);
	    count_disp++;
        }
    }
}

function func(i = 0) {
    using a = loggy("a");
    cont1: while (i-- > 0)
    {
        using b = loggy("b");
	let j = 3;
	while (j-- > 0) {
		using c = loggy("c");
		continue cont1;
	}
    }

    // Unreachable.
    // Never created, never disposed.
    using f = loggy("f");
}

function main()
{
	func(3);

	assert(count_created == count_disp, "not equal create-dispose");

	print("done.");
}