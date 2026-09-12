// Narrowing a discriminated union in a `switch` must not touch the payload until the
// discriminant has been compared. This program is ordinary and well typed - the only thing
// that makes it interesting is that the case carrying the LARGEST member is tested first,
// so a value of the smallest member reaches that test without matching an earlier case.
//
// The narrowing used to be emitted into the condition block, which reinterpreted the payload
// as NetworkSuccessState and retained it - and a union carries its members in a slot sized
// for the largest, so everything above a smaller member is uninitialized. Under `-mm=rc`
// those bytes were walked as string pointers and the program faulted before printing.
//
// The member tested first has to hold references for this to bite: a bigger member made only
// of numbers has nothing for the reference counting to walk.

type NetworkLoadingState = {
    state: "loading";
};

type NetworkFailedState = {
    state: "failed";
    code: number;
};

type NetworkSuccessState = {
    state: "success";
    response: {
        title: string;
        summary: string;
    };
};

type NetworkState =
    | NetworkLoadingState
    | NetworkFailedState
    | NetworkSuccessState;

function logger(state: NetworkState): string {
    switch (state.state) {
        case "success":
            return `Downloaded ${state.response.title} - ${state.response.summary}`;
        case "failed":
            return `Error ${state.code} downloading`;
        case "loading":
            return "Downloading...";
        default:
            return "<error>";
    }
}

function main() {
    assert(logger({ state: "loading" }) == "Downloading...");
    assert(logger({ state: "failed", code: 1.0 }) == "Error 1 downloading");
    assert(logger({ state: "success", response: { title: "title", summary: "summary" } }) == "Downloaded title - summary");

    print("done.");
}
