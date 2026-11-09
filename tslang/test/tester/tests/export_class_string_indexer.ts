namespace M {

    // A class with a string indexer defined in the exporting module; the importer
    // reaches it with constant keys (`bag["name"]`) as well as variable ones.
    // Cross-module counterpart to 00class_indexer_string_key.ts.

    export class Bag {
        private keys: string[] = [];
        private values: string[] = [];

        [key: string]: string;

        get(key: string): string {
            for (let i = 0; i < this.keys.length; i++) {
                if (this.keys[i] == key) {
                    return this.values[i];
                }
            }

            return "";
        }

        set(key: string, value: string) {
            let position = -1;
            for (let i = 0; i < this.keys.length; i++) {
                if (this.keys[i] == key) {
                    position = i;
                }
            }

            if (position < 0) {
                position = this.keys.length;
                this.keys.push(key + "");
                this.values.length = position + 1;
            }

            // in one place: under -mm=own a parameter is taken on every path or none
            this.values[position] = value;
        }

        get size() {
            return this.keys.length;
        }
    }
}
