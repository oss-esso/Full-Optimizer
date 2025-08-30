# Refactoring Comprehensive Integration Test

The file `@tests/comprehensive_integration_test.py` contains a lot of functions that can be added to `@algo//**` in the relevant files to make them part of the heuristic and be used in the future tests without having to import them from the comprehensive test.

## Instructions

1.  Identify a function or a small group of related functions in `tests/comprehensive_integration_test.py` that are suitable for moving to the `algo` directory.
2.  Determine the appropriate file within the `algo` directory to move the function(s) to.
3.  Move the function(s) to the target file.
4.  Update any necessary imports in `tests/comprehensive_integration_test.py` and any other files that might be affected.
5.  Run the comprehensive integration test (`tests/comprehensive_integration_test.py`) and ensure that all tests pass and the output is not messed up.
6.  Repeat this process for a couple of functions at a time until the refactoring is complete.
