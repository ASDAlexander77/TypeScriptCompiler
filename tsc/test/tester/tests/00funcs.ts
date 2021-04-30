function f1() {
    print("Hello World!");
}

function run(f: () => void) {
    f();
}

function main() {
    const x = f1;
    x();
    run(x);

	(function () {
		print("Hello World!");
	})();
}
