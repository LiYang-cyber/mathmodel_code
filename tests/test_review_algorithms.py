import numpy as np
import pandas as pd

from mathmodel.evaluation import entropy_weights, grey_relational_grade, orient_indicators, topsis
from mathmodel.forecasting import GM11
from mathmodel.optimization import particle_swarm


def test_entropy_topsis_ranks_dominant_alternative_first():
    frame = pd.DataFrame({"benefit": [10, 8, 4], "cost": [1, 3, 8]}, index=["A", "B", "C"])
    oriented = orient_indicators(frame, negative=["cost"])
    weights = entropy_weights(oriented)
    result = topsis(oriented, weights)
    assert np.isclose(weights.sum(), 1)
    assert result.loc["A", "rank"] == 1
    assert result.loc["A", "score"] > result.loc["C", "score"]


def test_grey_relation_identifies_matching_sequence():
    frame = pd.DataFrame({"reference": [1, 2, 4, 7], "same": [2, 4, 8, 14], "reverse": [7, 4, 2, 1]})
    grades = grey_relational_grade(frame, "reference")
    assert grades.index[0] == "same"


def test_gm11_forecasts_exponential_sequence():
    model = GM11().fit([2, 3, 4.5, 6.75, 10.125, 15.1875])
    forecast = model.predict(2)
    assert len(forecast) == 2
    assert forecast[1] > forecast[0] > 0


def test_particle_swarm_minimizes_sphere():
    result = particle_swarm(lambda x: float(np.sum(x**2)), [(-5, 5), (-5, 5)], particles=25,
                            iterations=80, seed=7)
    assert result.success
    assert result.fun < 1e-4
    assert len(result.history) == 81
