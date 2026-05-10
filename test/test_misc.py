import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from .helpers import tree_allclose


def test_fill_forward():
    in_ = jnp.array([jnp.nan, 0.0, 1.0, jnp.nan, jnp.nan, 2.0, jnp.nan])
    out_ = jnp.array([jnp.nan, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    fill_in = diffrax._misc.fill_forward(in_[:, None])
    assert tree_allclose(fill_in, out_[:, None], equal_nan=True)


def test_force_bitcast_convert_type():
    val_1 = jnp.float64(1e6)
    val_2 = jnp.float64(1e6 + 1e-4)

    # Val_1 and val_2 are different as float64,
    # but would be the same if naively downcast to float32.
    assert val_1 != val_2
    assert val_1.astype(jnp.int32) == val_2.astype(jnp.int32)

    val_1_cast = diffrax._misc.force_bitcast_convert_type(val_1, jnp.int32)
    val_2_cast = diffrax._misc.force_bitcast_convert_type(val_2, jnp.int32)

    assert val_1_cast.dtype == jnp.int32
    assert val_2_cast.dtype == jnp.int32

    # Bitcasted values should be different in the smaller type
    assert val_1_cast != val_2_cast


# ---------------------------------------------------------------------------
# Tests for AbstractImplicitSolver._residual_tags
# ---------------------------------------------------------------------------

# Use a concrete implicit solver to call the inherited method.
# negate_J=True  → DIRK (I - c·J), negate_J=False → ImplicitEuler (h·J - I)
_dirk = diffrax.Kvaerno3()
_impl = diffrax.ImplicitEuler()
# y_struct must have size > 1; lineax unconditionally treats size-1 operators as diagonal.
_y_struct = jax.ShapeDtypeStruct((5,), jnp.float64)


@pytest.mark.parametrize(
    "input_tags, negate_J, expected_tags",
    [
        # nsd_tag: DIRK (I - J) residual is PSD+symmetric; ImplEuler (J - I) is NSD+symmetric
        (
            frozenset({lx.negative_semidefinite_tag}),
            True,
            frozenset({lx.positive_semidefinite_tag, lx.symmetric_tag}),
        ),
        (
            frozenset({lx.negative_semidefinite_tag}),
            False,
            frozenset({lx.negative_semidefinite_tag, lx.symmetric_tag}),
        ),
        # psd_tag: dropped from residual (definiteness depends on step size); symmetric preserved
        (
            frozenset({lx.positive_semidefinite_tag}),
            True,
            frozenset({lx.symmetric_tag}),
        ),
        (
            frozenset({lx.positive_semidefinite_tag}),
            False,
            frozenset({lx.symmetric_tag}),
        ),
        # unit_diagonal_tag: dropped for both (I - J and J - I don't have unit diagonal)
        (
            frozenset({lx.unit_diagonal_tag}),
            True,
            frozenset(),
        ),
        (
            frozenset({lx.unit_diagonal_tag}),
            False,
            frozenset(),
        ),
        # diagonal_tag: preserved for both; identity is also diagonal/tridiagonal → symmetric too
        (
            frozenset({lx.diagonal_tag}),
            True,
            frozenset({lx.diagonal_tag, lx.symmetric_tag, lx.tridiagonal_tag}),
        ),
        (
            frozenset({lx.diagonal_tag}),
            False,
            frozenset({lx.diagonal_tag, lx.symmetric_tag, lx.tridiagonal_tag}),
        ),
        # symmetric_tag: preserved for both
        (
            frozenset({lx.symmetric_tag}),
            True,
            frozenset({lx.symmetric_tag}),
        ),
        (
            frozenset({lx.symmetric_tag}),
            False,
            frozenset({lx.symmetric_tag}),
        ),
        # tridiagonal_tag: preserved for both
        (
            frozenset({lx.tridiagonal_tag}),
            True,
            frozenset({lx.tridiagonal_tag}),
        ),
        (
            frozenset({lx.tridiagonal_tag}),
            False,
            frozenset({lx.tridiagonal_tag}),
        ),
        # empty: returns empty
        (frozenset(), True, frozenset()),
        (frozenset(), False, frozenset()),
    ],
)
def test_residual_tags(input_tags, negate_J, expected_tags):
    solver = _dirk if negate_J else _impl
    result = solver._residual_tags(input_tags, _y_struct, negate_J=negate_J)
    assert result == expected_tags, (
        f"input={input_tags!r}, negate_J={negate_J}: "
        f"got {result!r}, expected {expected_tags!r}"
    )
