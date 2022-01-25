#!/usr/bin/env python3

import sys
import re
import os

testCaseRegex = re.compile(r"TEST\s*\((\w+),\s*(\w+)\)")
testFixtureRegex = re.compile(r"TEST_F\s*\((\w+),\s*(\w+)\)")
expectRegex = re.compile(r"EXPECT_(\w+)\s*\((.*)\);")
assertRegex = re.compile(r"ASSERT_(\w+)\s*\((.*)\);")
leadingWhitespaceRegex = re.compile(r"^([\s/]*)")

module = "[unknown]"


def IsShiftOrComparison(c, i, check):
    if c in {"<", ">"}:
        # we can be the second or first character
        # look ahead first, could be = as well there
        if (i + 1) < len(check):
            if check[i + 1] in {"<", ">", "="}:
                return True

        # look behind, this is only needed for >> and <<
        if i > 0:
            if check[i - 1] == c:
                return True

        # there's also ->
        if c == ">":
            if check[i - 1] == "-":
                return True
    return False


def SplitCheck(check):

    # we start from the left, increment when we see anything of {, <, (,
    # decrement if we see }, >, ), and if we're at level 0 and see a , -- we
    # split at that location
    nesting = 0
    inString = False
    for i, c in enumerate(check):
        # Handle strings - we ignore everything inside quotes, but we still need
        # to check for escaped quotes
        # This handles "\"" and also correctly ignores "\\"
        if c == '"':
            if not inString:
                inString = True
            else:
                # escaped quote inside a string - ignore
                if i > 0 and check[i - 1] == "\\":
                    # Check again if the backslash was escaped
                    if i > 1 and check[i - 2] != "\\":
                        inString = False
                    else:
                        continue
                inString = False
            continue

        if inString:
            continue

        # Next problem case are shift and comparison operators, we might have
        # <=, <<, >=, >>, <, >. Unfortunately, we have no hope in figuring out
        # whether something like a<b,c> means a<b and c or a<b,c> with b,c being
        # template parameters. Thus, we're only going to handle the simple cases
        # for now, which are <=, <<, >=, >>

        if IsShiftOrComparison(c, i, check):
            continue

        if c in {"{", "<", "("}:
            nesting += 1
        elif c in {"}", ">", ")"}:
            nesting -= 1
        elif c == ",":
            if nesting > 0:
                continue
            else:
                return check[:i].strip(), check[i + 1 :].strip()
    else:
        return "/* GTEST-CATCH FIXME */" + check, ""


def ProcessCheck(match, prefix, line):
    basicComparisonOps = {
        "EQ": "==",
        "NE": "!=",
        "LE": ">=",
        "LT": ">",
        "GT": "<",
        "GE": "<=",
    }
    floatComparisonOps = {
        "FLOAT_EQ": "==",
        "FLOAT_NE": "!=",
        "DOUBLE_EQ": "==",
        "DOUBLE_NE": "!=",
        "NEAR": "==",
    }

    if match[0] == "TRUE":
        return prefix + (" ({});\n".format(match[1]))
    elif match[0] == "FALSE":
        return prefix + (" (!{});\n".format(match[1]))
    elif match[0] == "THAT":
        res, mat = SplitCheck(match[1])
        mat = mat.replace("HasSubstr", "Contains")
        if mat.startswith("Not("):
            mat = "!" + mat[4:-1]
        return prefix + f"_THAT ({res}, {mat});\n"
    elif match[0] in basicComparisonOps:
        expected, actual = SplitCheck(match[1])
        # LE/LT/GT/GE are turned around as EXPECT_LE(a, b) means a <= b, but we
        # want this to become b >= a
        op = basicComparisonOps[match[0]]
        return prefix + " ({} {} {});\n".format(actual, op, expected)
    elif match[0] in floatComparisonOps:
        expected, actual = SplitCheck(match[1])
        op = floatComparisonOps[match[0]]
        return prefix + " ({} {} Approx ({}));\n".format(actual, op, expected)
    elif match[0] == "THROW":
        expected, exception = SplitCheck(match[1])
        return prefix + "_THROWS_AS ({}, {});\n".format(expected, exception)
    elif match[0] == "NO_THROW":
        return prefix + "_NOTHROW ({});\n".format(match[1])
    elif match[0] == "STREQ":
        expected, actual = SplitCheck(match[1])
        return prefix + "_THAT ({}, Catch::Equals ({}));\n".format(actual, expected)
    else:
        raise Exception('Unknown match: "{}"'.format(line))


def ProcessLine(line):
    ws = leadingWhitespaceRegex.match(line).group(0)

    if "using ::testing::HasSubstr" in line:
        return line.replace("::testing::HasSubstr", "Catch::Matchers::Contains")
    elif "using ::testing::Not" in line:
        return "\n"
    elif '#include "gmock/gmock.h"' in line:
        return "\n"

    match = testCaseRegex.search(line)
    if match:
        g0, g1 = match.groups()
        return ws + f'TEST_CASE("{g0}: {g1}", "{module}") {{\n'

    match = testFixtureRegex.search(line)
    if match:
        app, g1 = match.groups()
        return ws + f'TEST_CASE_METHOD({app}, "{g1}", "{module}") {{\n'

    match = expectRegex.search(line)
    if match:
        g = match.groups()
        return ProcessCheck(g, ws + "CHECK", line)

    match = assertRegex.search(line)
    if match:
        g = match.groups()
        return ProcessCheck(g, ws + "REQUIRE", line)

    return line


if __name__ == "__main__":
    module = "[{}]".format(sys.argv[1])
    for p in sys.argv[2:]:
        with open(p, "r") as f:
            lines = f.readlines()
        output = map(ProcessLine, lines)
        with open(p, "w") as f:
            f.write("".join(output))
