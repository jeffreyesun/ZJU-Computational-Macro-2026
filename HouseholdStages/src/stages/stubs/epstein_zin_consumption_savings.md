This is just like consumption-savings, except that utility and V are aggregated according to:
$$
    V^\text{start} = \left[ (1-\beta)\, C_t^{\frac{\sigma-1}{\sigma}} + \beta \left( \mathbb{E}_t \left[ V_{t+1}^\text{end} \right] \right)^{\frac{1}{1-\alpha} \cdot \frac{\sigma-1}{\sigma}} \right]^{\frac{\sigma}{\sigma-1}}.
$$

This probably needs to be two stages. One for the EoS-style aggregation, and one for taking the power of V to $\alpha$ which is inserted after shocks take place\dots.

It would also be nice to do the time discounting in the TimeDiscounting stage, with the caveat that the time-discounting factor would need to be taken to the power of $\frac{1}{1-\alpha} \cdot \frac{\sigma-1}{\sigma}$, or something.

Another option would be to rewrite in terms of $\tilde{V} \equiv V^{1-\alpha}$,
$$
\tilde{V}_t = \left[ (1-\beta)\, C_t^{\frac{\sigma-1}{\sigma}} + \beta \left( \mathbb{E}_t [\tilde{V}_{t+1}] \right)^{\frac{1}{1-\alpha} \cdot \frac{\sigma-1}{\sigma}} \right]^{\frac{(1-\alpha)\sigma}{\sigma-1}}
$$

Not sure of best way right now.
