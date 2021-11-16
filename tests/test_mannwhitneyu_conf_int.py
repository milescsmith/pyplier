import hypothesis.strategies as st
from hypothesis import assume, given

from pyplier.AUC import mannwhitneyu_conf_int


@given(
    x=st.lists(elements=st.integers(), min_size=20),
    y=st.lists(elements=st.integers(), min_size=20),
)
def testconfint(x, y):
    assume(len(x) > 20)
    assume(len(y) > 20)
    mannwhitneyu_conf_int(x, y)
