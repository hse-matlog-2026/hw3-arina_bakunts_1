# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: propositions/syntax.py

"""Syntactic handling of propositional formulas."""

from __future__ import annotations
from functools import lru_cache
from typing import Mapping, Optional, Set, Tuple, Union

from logic_utils import frozen, memoized_parameterless_method

@lru_cache(maxsize=100) # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= 'p' and string[0] <= 'z' and \
           (len(string) == 1 or string[1:].isdecimal())

@lru_cache(maxsize=100) # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant, ``False`` otherwise.
    """
    return string == 'T' or string == 'F'

@lru_cache(maxsize=100) # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'

@lru_cache(maxsize=100) # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    # For Chapter 3:
    return string in {'&', '|',  '->', '+', '<->', '-&', '-|'}

@frozen
class Formula:
    """An immutable propositional formula in tree representation, composed from
    variable names, and operators applied to them.

    Attributes:
        root (`str`): the constant, variable name, or operator at the root of
            the formula tree.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
    """
    root: str
    first: Optional[Formula]
    second: Optional[Formula]

    def __init__(self, root: str, first: Optional[Formula] = None,
                 second: Optional[Formula] = None):
        """Initializes a `Formula` from its root and root operands.

        Parameters:
            root: the root for the formula tree.
            first: the first operand for the root, if the root is a unary or
                binary operator.
            second: the second operand for the root, if the root is a binary
                operator.
        """
        if is_variable(root) or is_constant(root):
            assert first is None and second is None
            self.root = root
        elif is_unary(root):
            assert first is not None and second is None
            self.root, self.first = root, first
        else:
            assert is_binary(root)
            assert first is not None and second is not None
            self.root, self.first, self.second = root, first, second

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        # Task 1.1
        if is_variable(self.root) or is_constant(self.root):
            return self.root
        if is_unary(self.root):
            return self.root + str(self.first)
        if is_binary(self.root):
            return '(' + str(self.first) + self.root + str(self.second) + ')'

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @memoized_parameterless_method
    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        # Task 1.2
        result = set()
        if is_variable(self.root):
            result.add(self.root)
        elif is_unary(self.root):
            result = result.union(self.first.variables())
        elif is_binary(self.root):
            result = result.union(self.first.variables()).union(self.second.variables())
        return result

    @memoized_parameterless_method
    def operators(self) -> Set[str]:
        """Finds all operators in the current formula.

        Returns:
            A set of all operators (including ``'T'`` and ``'F'``) used in the
            current formula.
        """
        # Task 1.3
        result = set()
        if is_constant(self.root):
            result.add(self.root)
        elif is_unary(self.root):
            result = result.union(self.root).union(self.first.operators())
        elif is_binary(self.root):
            result.add(self.root)
            result = result.union(self.first.operators()).union(self.second.operators())
        return result

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Union[Formula, None], str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a variable name (e.g.,
            ``'x12'``) or a unary operator followed by a variable name, then the
            parsed prefix will include that entire variable name (and not just a
            part of it, such as ``'x1'``). If no prefix of the given string is a
            valid standard string representation of a formula then returned pair
            should be of ``None`` and an error message, where the error message
            is a string with some human-readable content.
        """
        # Task 1.4
        if len(string) == 0:
            return None, ''
        if is_constant(string[0]):
            return Formula(string[0]), string[1:]
        if is_variable(string[0]):
            i = 0
            prefix = string[0]
            while is_variable(prefix):
                i += 1
                if i == len(string):
                    return Formula(prefix), ''
                prefix += string[i]
            return Formula(prefix[:-1]), string[i:]
        if is_unary(string[0]):
            prefix, suffix = Formula._parse_prefix(string[1:])
            if prefix is None:
                return None, string
            return Formula(string[0], prefix), suffix
        if string[0] == '(':
            prefix, suffix = Formula._parse_prefix(string[1:])
            if suffix == '':
                return None, string
            if suffix[0] not in {'-','<'}:
                a, b = Formula._parse_prefix(suffix[1:])
                if a is None or not is_binary(suffix[0]) or len(b) == 0 or b[0] != ')':
                    return None, string
                return Formula(suffix[0], prefix, a), b[1:]
            elif suffix[0] == '-':
                a, b = Formula._parse_prefix(suffix[2:])
                if a is None or not is_binary(suffix[:2]) or len(b) == 0 or b[0] != ')':
                    return None, string
                return Formula(suffix[:2], prefix, a), b[1:]
            else:
                a, b = Formula._parse_prefix(suffix[3:])
                if a is None or not is_binary(suffix[:3]) or len(b) == 0 or b[0] != ')':
                    return None, string
                return Formula(suffix[:3], prefix, a), b[1:]
        return None, string

    @staticmethod
    def is_formula(string: str) -> bool:
        """Checks if the given string is a valid representation of a formula.

        Parameters:
            string: string to check.

        Returns:
            ``True`` if the given string is a valid standard string
            representation of a formula, ``False`` otherwise.
        """
        # Task 1.5
        f, suff = Formula._parse_prefix(string)
        return f is not None and suff == ''
        
    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        assert Formula.is_formula(string)
        # Task 1.6
        if is_constant(string[0]):
            return Formula(string)
        if is_unary(string[0]):
            return Formula(string[0], Formula.parse(string[1:]))
        if is_variable(string[0]):
            i = 0
            v = ''
            while i < len(string):
                v += string[i]
                i += 1
            return Formula(v)
        if string[0] == '(':
            i = 1
            n_open = 0
            while not ((is_binary(string[i]) or string[i] in {'-','<'}) and n_open == 0):
                if string[i] == '(':
                    n_open += 1
                elif string[i] == ')':
                    n_open -= 1
                i += 1
            if string[i] == '-':
                return Formula(string[i:i + 2], Formula.parse(string[1:i]), Formula.parse(string[i + 2:-1]))
            elif string[i] == '<':
                return Formula(string[i:i + 3], Formula.parse(string[1:i]), Formula.parse(string[i + 3:-1]))
            return Formula(string[i], Formula.parse(string[1:i]), Formula.parse(string[i + 1:-1]))

    def polish(self) -> str:
        """Computes the polish notation representation of the current formula.

        Returns:
            The polish notation representation of the current formula.
        """
        # Optional Task 1.7
        if is_unary(self.root):
            return '~' + self.first.polish()
        if is_binary(self.root):
            return self.root + self.first.polish() + self.second.polish()
        if is_variable(self.root) or is_constant(self.root):
            return self.root

    @staticmethod
    def parse_polish(string: str) -> Formula:
        """Parses the given polish notation representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose polish notation representation is the given string.
        """
        # Optional Task 1.8
        if is_unary(string[0]):
            return Formula(string[0], Formula.parse_polish(string[1:]))
        if is_variable(string[0]) or is_constant(string[0]):
            return Formula(string)
        stack = []
        i = len(string) - 1
        var = ''
        while i >= 0:
            if string[i].isdecimal():
                var = string[i] + var
            if is_variable(string[i]):
                stack.append(Formula(string[i] + var))
                var = ''
            if is_constant(string[i]):
                stack.append(Formula(string[i]))
            if is_unary(string[i]):
                stack.append(Formula(string[i], stack.pop()))
            if is_binary(string[i]):
                stack.append(Formula(string[i], stack.pop(), stack.pop()))
            if string[i] == '>':
                stack.append(Formula('->', stack.pop(), stack.pop()))
                i -= 1
            i -= 1
        return stack[0]

    def substitute_variables(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each variable name `v` that is a
        key in `substitution_map` with the formula `substitution_map[v]`.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            variable name occurrences originating in the current formula are
            substituted (i.e., variable name occurrences originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((p->p)|r)').substitute_variables(
            ...     {'p': Formula.parse('(q&r)'), 'r': Formula.parse('p')})
            (((q&r)->(q&r))|p)
        """
        for variable in substitution_map:
            assert is_variable(variable)
        # Task 3.3
        if is_constant(self.root):
            return self
        if is_variable(self.root):
            if self.root in substitution_map.keys():
                return substitution_map[self.root]
            return self
        if is_unary(self.root):
            return Formula(self.root, Formula.substitute_variables(self.first, substitution_map))
        return Formula(self.root, Formula.substitute_variables(self.first, substitution_map), Formula.substitute_variables(self.second, substitution_map))

    def substitute_operators(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each constant or operator `op`
        that is a key in `substitution_map` with the formula
        `substitution_map[op]` applied to its (zero or one or two) operands,
        where the first operand is used for every occurrence of ``'p'`` in the
        formula and the second for every occurrence of ``'q'``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            operator occurrences originating in the current formula are
            substituted (i.e., operator occurrences originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((x&y)&~z)').substitute_operators(
            ...     {'&': Formula.parse('~(~p|~q)')})
            ~(~~(~x|~y)|~~z)
        """
        for operator in substitution_map:
            assert is_constant(operator) or is_unary(operator) or \
                   is_binary(operator)
            assert substitution_map[operator].variables().issubset({'p', 'q'})
        # Task 3.4
        if is_constant(self.root):
            if self.root in substitution_map.keys():
                return substitution_map[self.root]
            return self
        if is_unary(self.root):
            if self.root in substitution_map.keys():
                return substitution_map[self.root].substitute_variables({'p': self.first.substitute_operators(substitution_map)})
            return Formula(self.root, self.first.substitute_operators(substitution_map))
        if is_binary(self.root):
            if self.root in substitution_map.keys():
                return substitution_map[self.root].substitute_variables({'p': self.first.substitute_operators(substitution_map), 'q': self.second.substitute_operators(substitution_map)})
            return Formula(self.root, self.first.substitute_operators(substitution_map), self.second.substitute_operators(substitution_map))
        return self
