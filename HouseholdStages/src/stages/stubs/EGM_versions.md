It would be nice if stage `Spec`s were fully agnostic as to implementation. Then the current Buffer objects could contain the layout information and implementation specification.

In particular, this would allow the same Spec to be used for endogenous-gridpoints or Reiter-style implementations.

Another possibly more modular alternative would be to just have EGMLogitStage, EGMProductStage, etc., that are totally separate implementations and files.
