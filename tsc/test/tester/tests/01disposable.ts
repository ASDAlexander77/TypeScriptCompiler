let count = 0;

function loggy(id: string) {
    print(`Creating ${id}`);
    return {
        [Symbol.dispose]() {
            print(`Disposing ${id}`);
	    count++;
        }
    }
}

function func() {
    using a = loggy("a");
    using b = loggy("b");
    {
        using c = loggy("c");
        using d = loggy("d");
    }
    using e = loggy("e");
    return;
    // Unreachable.
    // Never created, never disposed.
    using f = loggy("f");
}

function main()
{
	func();
	
	assert(count == 5, "Not all 'dispose' called");

	print("done.");
}