function main() {
    type StaffAccount = [number, string, string, string?];

    const staff: StaffAccount[] = [
        [0, "Adankwo", "adankwo.e@"],
        [1, "Kanokwan", "kanokwan.s@"],
        [2, "Aneurin", "aneurin.s@", "Supervisor"],
    ];

    //assert(staff[0][3] == "Supervisor");

    print("done.");
}