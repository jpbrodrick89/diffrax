from collections.abc import Callable
from typing import ClassVar

import equinox.internal as eqxi
import numpy as np
import optimistix as optx

from .._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .._root_finder import VeryChord, with_stepsize_controller_tols
from .runge_kutta import AbstractESDIRK, ButcherTableau


# TR-BDF2 is usually presented as a composite step: a trapezoidal-rule sub-step from
# t0 to t0 + γh, followed by a BDF2 sub-step (using y0, the trapezoidal value, and the
# unknown y1) from t0 + γh to t0 + h. Substituting the trapezoidal sub-step into the
# BDF2 formula collapses this into an ordinary 3-stage ESDIRK tableau -- see
# Hosea and Shampine, "Analysis and implementation of TR-BDF2" (1996).
#
# γ = 2 - √2 is the classical choice (Bank et al. 1985) that makes the method
# L-stable. It has the pleasant side effect of making the trapezoidal sub-stage and
# the BDF2 sub-stage share the same diagonal coefficient γ/2, which is exactly what
# `AbstractESDIRK` needs in order to reuse a single Newton Jacobian across both
# implicit stages.
γ = 2 - np.sqrt(2)
d = γ / 2
a21 = d
a31 = 1 / (2 * np.sqrt(2))
a32 = a31
# See /devdocs/predictor_dirk.md
α21 = 1.0
α31 = 1 - 1 / γ
α32 = 1 / γ

_tr_bdf2_tableau = ButcherTableau(
    a_lower=(
        np.array([a21]),
        np.array([a31, a32]),
    ),
    a_predictor=(np.array([α21]), np.array([α31, α32])),
    a_diagonal=np.array([0, d, d]),
    b_sol=np.array([a31, a32, d]),
    b_error=np.array([(1 - γ) / 3, -1 / 3, γ / 3]),
    c=np.array([γ, 1.0]),
)


class TRBDF2(AbstractESDIRK):
    r"""TR-BDF2 method.

    A-L stable stiffly accurate 2nd order ESDIRK method. Has an embedded 3rd order
    accurate error estimate for adaptive step sizing. Uses 3 stages with FSAL. Uses
    3rd order Hermite interpolation for dense/ts output.

    Each step is composed of a trapezoidal-rule sub-step followed by a BDF2 sub-step,
    which combine to give an L-stable method that is cheaper per step than the
    higher-order [`diffrax.Kvaerno3`][]/[`diffrax.Kvaerno4`][]/[`diffrax.Kvaerno5`][]
    family (2 implicit solves per step, rather than 3, 5, or 6), at the cost of lower
    (2nd) order accuracy.

    ??? cite "References"

        ```bibtex
        @article{bank1985trbdf2,
          title={Transient simulation of silicon devices and circuits},
          author={Bank, Randolph E and Coughran, William M and Fichtner, Wolfgang
                  and Grosse, Eric H and Rose, Donald J and Smith, Robert K},
          journal={IEEE Transactions on Electron Devices},
          volume={32},
          number={10},
          pages={1992--2007},
          year={1985},
          publisher={IEEE}
        }

        @article{hosea1996analysis,
          title={Analysis and implementation of TR-BDF2},
          author={Hosea, ME and Shampine, LF},
          journal={Applied Numerical Mathematics},
          volume={20},
          number={1-2},
          pages={21--37},
          year={1996},
          publisher={Elsevier}
        }
        ```
    """

    tableau: ClassVar[ButcherTableau] = _tr_bdf2_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 10

    def order(self, terms):
        del terms
        return 2


eqxi.doc_remove_args("scan_kind")(TRBDF2.__init__)
TRBDF2.__init__.__doc__ = """**Arguments:**

- `root_finder`: an [Optimistix](https://github.com/patrick-kidger/optimistix) root
    finder to solve the implicit problem at each stage.
- `root_find_max_steps`: the maximum number of steps that the root finder is allowed to
    make before unconditionally rejecting the step. (And trying again with whatever
    smaller step that adaptive stepsize controller proposes.)
"""
