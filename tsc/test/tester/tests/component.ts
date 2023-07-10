import Service from "./service";

class MyComponent {
    constructor(public Service: Service) {
    }

    method(x: this) {
    }
}

function main()
{
	const myComponent = new MyComponent(null);
	print("done.");
}