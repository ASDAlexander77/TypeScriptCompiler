// Link 3 of the chain deep_chain.ts imports; see there.
import './chain_02'
import './chain_01'

export function chain3() { return chain2() + 3; }
