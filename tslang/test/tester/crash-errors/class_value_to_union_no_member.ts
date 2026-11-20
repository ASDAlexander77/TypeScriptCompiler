class Store { x = 1; }
interface A { prototype: Store; }
interface B { z: number; }
function inject(token: A | B) { }
inject(Store);
