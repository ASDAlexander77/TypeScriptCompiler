class B { readonly k: number = 1; }
class D extends B { constructor() { super(); this.k = 3; } }
function main() { print(new D().k); }
