function main() {
    type StaffAccount = [number, string, string, string?];

    const staff: StaffAccount[] = [
        [0, "Adankwo", "adankwo.e@"],
        [1, "Kanokwan", "kanokwan.s@"],
        [2, "Aneurin", "aneurin.s@", "Supervisor"],
    ];

    for (const v of staff) print(v[0], v[1], v[2], v[3] || "<no value>");

    assert(staff[2][3] == "Supervisor");

    print("done.");
}