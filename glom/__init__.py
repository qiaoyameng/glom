from glom.core import (glom,
                       Fill,
                       Auto,
                       register,
                       register_op,
                       Glommer,
                       Call,
                       Invoke,
                       Spec,
                       Ref,
                       SKIP,
                       STOP,
                       UP,
                       ROOT,
                       MODE,
                       Path,
                       Vars,
                       Val,
                       Coalesce,
                       Inspect,
                       Pipe,
                       GlomError,
                       BadSpec,
                       PathAccessError,
                       PathAssignError,
                       CoalesceError,
                       UnregisteredTarget,
                       T, S, A)

from glom.reduction import Sum, Fold, Flatten, flatten, FoldError, Merge, merge
from glom.matching import (M,
                           Or,
                           And,
                           Not,
                           Match,
                           MatchError,
                           TypeMatchError,
                           Regex,
                           Optional,
                           Required,
                           Switch,
                           Check,
                           CheckError)
from glom.mutation import Assign, Delete, assign, delete, PathDeleteError

# there's no -ion word that really fits what "streaming" means.
# generation, production, iteration, all have more relevant meanings
# elsewhere. (maybe procrastination :P)
from glom.streaming import Iter

# budgeted grouping that spills inactive groups to a local SQLite store
from glom.spill import (SpillGroup, SpillableAgg, SpillError,
                       SpillCount, SpillSum, SpillMin, SpillMax)

from glom._version import __version__
