function foo(x: number, y: number, ...z: string[]) {
	print("foo: ", "x:", x, "\ty:", y, "\tz[0]:", z.length > 0 ? z[0] : "<empty-0>", "\tz[1]:", z.length > 1 ? z[1] : "<empty-1>");
}

function main() {

    foo(...[1, 2, "abc", "Can u see me?"]);

    print("done.");
}