A stage that advances the age of the household by one period in an OLG model. It would include the bequest-utility function (closure) for the backwards pass and the entry function (closure) for the forwards pass. Otherwise it would just roll everything down by one along a specified dimension.

Ideally, this would be implemented as Utility $\circ$ Markov $\circ$ Entry, where the Markov stage is just a shifted Identity, rolling some people into nonexistence.
