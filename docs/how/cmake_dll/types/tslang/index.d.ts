
declare function print(...args: any[]) : void;
declare function assert(cond: boolean, msg?: string) : void;
declare type byte = any;
declare type short = any;
declare type ushort = any;
declare type int = any;
declare type uint = any;
declare type index = any;
declare type long = any;
declare type ulong = any;
declare type char = any;
declare type i8 = any;
declare type i16 = any;
declare type i32 = any;
declare type i64 = any;
declare type u8 = any;
declare type u16 = any;
declare type u32 = any;
declare type u64 = any;
declare type s8 = any;
declare type s16 = any;
declare type s32 = any;
declare type s64 = any;
declare type f16 = any;
declare type f32 = any;
declare type f64 = any;
declare type f128 = any;
declare type half = any;
declare type float = any;
declare type double = any;
declare type Opaque = any;

type Ref<T> = any
type Reference<T> = Ref<T> // deprecated alias of Ref

declare function Ref<T>(r: T): Ref<T>;
declare function Deref<T>(r: Ref<T>): T;

// deprecated aliases of Ref / Deref
declare function ReferenceOf<T>(r: T): Ref<T>;
declare function LoadReference<T>(r: Ref<T>): T;

declare function sizeof<T>(v?: T): index;
