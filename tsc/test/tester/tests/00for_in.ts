function main() {
    /* 
   const object : { name: string, age: number } = { name: "foo", age: 10.0 };

   for (const property in object) {
     print(`${property}: ${object[property]}`);
   }
	*/

    const object = [10, 20, 30];

    for (const property in object) {
        print(`${property}: ${object[property]}`);
    }

    print("done.");
}
