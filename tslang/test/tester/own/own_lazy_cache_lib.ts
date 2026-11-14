// TSLANG_ISSUES 47: the library own_lazy_cache_main imports - native_ts_graphics' cached getter, in
// an exported class in a namespace. An exported method says it returns a borrow of `this`
// (`@lends`), and `@final` that no override of it may overwrite the field (TSLANG 48).
export namespace Nri {
    export class Queue {
        constructor(public h: number) {}
    }

    export class Device {
        cachedQueue: Queue | undefined;
        made = 0;

        @final @lends("this") getGraphicsQueue(): Queue {
            let result = this.cachedQueue;
            if (!result) {
                this.made++;
                result = new Queue(7);
                this.cachedQueue = result;
            }

            return result;
        }
    }
}
