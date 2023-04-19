function main() {

    let obj = {
        val: 10,
        add: () => {
            add_();

            function add_() {
                if (++this.val < 15) this.add();
            }
        }
    };

    obj.add();
    print(obj.val);

    assert(obj.val === 15);

    print("done.");
}
