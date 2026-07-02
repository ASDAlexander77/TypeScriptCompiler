async function adder(a = 0, b = 0) {
    return a + b;
}

export class Adder {
	#a: int;
	#b: int;

	constructor(a: int, b: int) {
		this.#a = a;
		this.#b = b;
	}

	get result() { return await adder(this.#a, this.#b); }
}
