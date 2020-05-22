import sympy
import pint

from system.potential import AnalyticalPotential, ParametrizedAnalyticalPotential
from system.utils import compare_sympy_expr
from system.tests.base_test import BaseTest


u = pint.UnitRegistry()

class TestPotential(BaseTest):
    """Test behavior of Potential classes."""
    def test_analytical_potential_constructor(self):
        """Test that Potential objects can be constructed from a combination of types."""
        pot = AnalyticalPotential(
            name="TestPotential",
            expression=sympy.sympify("m*x+b"),
            independent_variables={sympy.sympify("x")},
        )

        assert pot.name == "TestPotential"
        assert compare_sympy_expr(pot.expression, "m*x+b")

        pot_from_str = AnalyticalPotential(
            name="TestPotentialFromString",
            expression="m*x+b",
            independent_variables={"x"},
        )

        assert compare_sympy_expr(pot.expression, pot_from_str.expression)

    def test_parametrized_analytical_potential_constructor(self):
        """Test that a ParametrizedAnalyticalPotential can be constructed."""
        pot = ParametrizedAnalyticalPotential(
            name="TestPotential",
            expression="m*x+b",
            independent_variables={"x"},
            parameters={"m": 0.5 * u.dimensionless, "b": -1.0 * u.dimensionless},
        )

        assert pot.name == "TestPotential"
        assert compare_sympy_expr(pot.expression, "m*x+b")
        assert "m" in pot.parameters.keys()
        assert "b" in pot.parameters.keys()
        assert pot.parameters["m"] == 0.5 * u.dimensionless
        assert pot.parameters["b"] == -1.0 * u.dimensionless


class TestClassIPotentials(BaseTest):
    """Test expressions specific to Class I force fields."""
    expressions = {
        'Lennard-Jones': '4*espilon*((sigma/r)**12-(sigma/r)*6)',
        'HarmonicBond': '0.5*k*(length_length_0)**2',
        'HarmonicAngle': '0.5*k*(theta-theta_0)**2',
        'HarmonicTorsion': '0.5*k*(phi-phi_0)**2',
        }

    independent_variables = {
        'Lennard-Jones': 'r',
        'HarmonicBond': 'length',
        'HarmonicAngle': 'theta',
        'HarmonicTorsion': 'phi',
    }

    def test_load_potentials(self):
        """Test that expressions in Class I force fields can be read."""
        for pot_type in self.expressions.keys():
            pot = AnalyticalPotential(
                name=pot_type,
                expression=self.expressions[pot_type],
                independent_variables=self.independent_variables[pot_type],
            )

            assert pot.expression == self.expressions[pot_type]
            assert pot.independent_variables == self.independent_variables[pot_type]
