function main() {
    const arr = [1, 2, 3, 4, 5];
    
    let count = 0;
    for (const v of arr.filter(x => x % 2))
    {
	print(v);
	count++;
    }

    assert(count == 3);

    print("done.");
}
