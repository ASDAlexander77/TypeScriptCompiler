interface IObservable<T> {
    n: IObservable<T[]>;
}
var values: IObservable<number>;
var n = values.n;
