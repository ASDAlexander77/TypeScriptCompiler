// Example source in TypeScript language.
// `tslang` compiler turns this into mycode.obj, which CMake links
// with main.cpp. Replace with real TypeScript syntax; the symbols exported
// must match the extern "C" declarations in main.cpp.

async function adder(a = 0, b = 0) {
    return a + b;
}

class Adder
{
	#a: int;
	#b: int;

	constructor(a: int, b: int) {
		this.#a = a;
		this.#b = b;
	}

	get result() { return await adder(this.#a, this.#b); }
}

export function foo_add(a: int, b: int): int {
    const adder = new Adder(a, b);
    return adder.result;
}

export function foo_hello() {
    console.log("hello from foo");
}
