function main() {
    const inventory = [
    { name: "asparagus", type: "vegetables", quantity: 5 },
    { name: "bananas", type: "fruit", quantity: 0 },
    { name: "goat", type: "meat", quantity: 23 },
    { name: "cherries", type: "fruit", quantity: 5 },
    { name: "fish", type: "meat", quantity: 22 },
    ];

    for (const v of inventory) print(v.name, v.type, v.quantity);

    for (const r of inventory.map(({ type }) => type)) print(r);

    print("done.");
}