interface Observable<T> {
    map<U>(proj: (el: T) => U): Observable<U>;
}

function main() {
    let o: Observable<number>;
    o.map<string>();
}