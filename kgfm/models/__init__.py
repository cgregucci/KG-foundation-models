"""Re-exports the config-dispatchable model classes.

Configs reference these classes by short name (`Ultra`, `MOTIF`,
`TRIXEntity`, `TRIXRelation`, `TRIXNoIter`); entry-point scripts dispatch
on `cfg.model["class"]`.
"""

from kgfm.models.ultra import Ultra, RelNBFNet, EntityNBFNet
from kgfm.models.motif import MOTIF, RelHCNet
from kgfm.models.trix_entity import TRIXEntity
from kgfm.models.trix_relation import TRIXRelation
from kgfm.models.trix_noiter import TRIXNoIter

__all__ = [
    "Ultra", "MOTIF", "TRIXEntity", "TRIXRelation", "TRIXNoIter",
    "RelNBFNet", "EntityNBFNet", "RelHCNet",
]
