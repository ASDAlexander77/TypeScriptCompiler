interface CreateArtistBioBase {
    artistID: string;
    thirdParty?: boolean;
}

type CreateArtistBioRequest = CreateArtistBioBase & ({ html: string } | { markdown: string });

const workingRequest: CreateArtistBioRequest = {
    artistID: "banksy",
    markdown: "Banksy is an anonymous England-based graffiti artist...",
};

function main() {
    print("done.");
}
