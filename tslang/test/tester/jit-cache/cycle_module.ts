import './cycle_main'

export function fromModule() {
    return "module+" + fromMain();
}
