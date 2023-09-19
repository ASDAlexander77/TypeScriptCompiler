import Service from "./service";

class MyComponent {
    constructor(public Service: Service) {
    }

    method(x: this) {
    }
}

function main()
{
	const myComponent = new MyComponent(new Service());
	print("done.");
}