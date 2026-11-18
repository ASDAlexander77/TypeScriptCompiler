type InfiniteArray<T> = InfiniteArray<T>[];
type Flatten<T> = T extends readonly (infer U)[] ? Flatten<U> : T;
type F = Flatten<InfiniteArray<string>>;
let f: F;
