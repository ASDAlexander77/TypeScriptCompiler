type Array<T> = T[];

function ForEach<T>(this: Array<T>, callbackfn: (value: T, index?: number, array?: Array<T>) => void, thisArg?: any): void 
{
    let index = 0;
    for (const val of this) callbackfn(val, index++, this);
}

function main() {
    [1, 2, 3].ForEach((x, y?, z?) => { print(x); });
    print("done.");
}
