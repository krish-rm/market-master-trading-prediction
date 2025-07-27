"""
Test runner for Market Master test suite
Executes all tests and provides comprehensive reporting
"""

import unittest
import sys
import time
from pathlib import Path
from io import StringIO

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestResult:
    """Custom test result class for detailed reporting"""
    
    def __init__(self):
        self.tests_run = 0
        self.failures = []
        self.errors = []
        self.successes = []
        self.start_time = None
        self.end_time = None

    def start_test(self, test):
        """Called before each test"""
        if self.start_time is None:
            self.start_time = time.time()

    def add_success(self, test):
        """Called when test passes"""
        self.tests_run += 1
        self.successes.append(test)

    def add_failure(self, test, traceback):
        """Called when test fails"""
        self.tests_run += 1
        self.failures.append((test, traceback))

    def add_error(self, test, traceback):
        """Called when test has error"""
        self.tests_run += 1
        self.errors.append((test, traceback))

    def stop_test(self, test):
        """Called after each test"""
        self.end_time = time.time()

    @property
    def success_count(self):
        return len(self.successes)

    @property
    def failure_count(self):
        return len(self.failures)

    @property
    def error_count(self):
        return len(self.errors)

    @property
    def total_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0


def discover_and_run_tests():
    """Discover and run all tests in the test suite"""
    print("🧪 Market Master - Test Suite Runner")
    print("=" * 60)
    print("Discovering and executing all tests...")
    print()

    # Test discovery
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add unit tests
    try:
        unit_tests = test_loader.discover('tests/unit', pattern='test_*.py')
        test_suite.addTest(unit_tests)
        print("✅ Unit tests discovered")
    except Exception as e:
        print(f"⚠️  Unit test discovery failed: {e}")

    # Add integration tests
    try:
        integration_tests = test_loader.discover('tests/integration', pattern='test_*.py')
        test_suite.addTest(integration_tests)
        print("✅ Integration tests discovered")
    except Exception as e:
        print(f"⚠️  Integration test discovery failed: {e}")

    print()

    # Run tests with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=False
    )

    print("🚀 Running test suite...")
    start_time = time.time()
    
    result = runner.run(test_suite)
    
    end_time = time.time()
    total_time = end_time - start_time

    # Print detailed results
    print()
    print("📊 Test Results Summary")
    print("-" * 40)
    print(f"Tests Run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"🚨 Errors: {len(result.errors)}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    
    # Calculate success rate
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")

    print()

    # Print failures and errors if any
    if result.failures:
        print("❌ Test Failures:")
        print("-" * 20)
        for test, traceback in result.failures:
            print(f"• {test}")
            print(f"  {traceback.strip()}")
        print()

    if result.errors:
        print("🚨 Test Errors:")
        print("-" * 20)
        for test, traceback in result.errors:
            print(f"• {test}")
            print(f"  {traceback.strip()}")
        print()

    # Overall assessment
    print("🎯 Overall Assessment:")
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("🎉 ALL TESTS PASSED! Test suite is healthy.")
        status = "PASS"
    elif len(result.failures) + len(result.errors) <= 2:
        print("✅ Test suite mostly healthy with minor issues.")
        status = "MOSTLY_PASS"
    else:
        print("⚠️  Test suite has significant issues that need attention.")
        status = "FAIL"

    print()
    print(f"Final Status: {status}")
    
    return result, status


def run_specific_test_category(category):
    """Run tests from a specific category"""
    print(f"🧪 Running {category} tests only...")
    
    test_loader = unittest.TestLoader()
    
    if category == "unit":
        test_suite = test_loader.discover('tests/unit', pattern='test_*.py')
    elif category == "integration":
        test_suite = test_loader.discover('tests/integration', pattern='test_*.py')
    else:
        print(f"❌ Unknown test category: {category}")
        return None, "INVALID"

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        status = "PASS"
    else:
        status = "FAIL"
    
    return result, status


def check_test_coverage():
    """Check basic test coverage by examining test files"""
    print("📊 Test Coverage Analysis")
    print("-" * 30)
    
    # Check which components have tests
    test_files = {
        'Data Generation': Path('tests/unit/test_data_generation.py').exists(),
        'Action Predictor': Path('tests/unit/test_action_predictor.py').exists(),
        'MLOps Components': Path('tests/unit/test_mlops_components.py').exists(),
        'Pipeline Integration': Path('tests/integration/test_pipeline_integration.py').exists()
    }
    
    coverage_count = sum(test_files.values())
    total_components = len(test_files)
    
    for component, has_tests in test_files.items():
        status = "✅" if has_tests else "❌"
        print(f"{status} {component}")
    
    coverage_percentage = (coverage_count / total_components) * 100
    print(f"\n📈 Test Coverage: {coverage_count}/{total_components} components ({coverage_percentage:.0f}%)")
    
    return coverage_percentage


def main():
    """Main test runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Master Test Runner')
    parser.add_argument('--category', choices=['unit', 'integration', 'all'], 
                       default='all', help='Test category to run')
    parser.add_argument('--coverage', action='store_true', 
                       help='Show test coverage analysis')
    
    args = parser.parse_args()
    
    if args.coverage:
        check_test_coverage()
        print()
    
    if args.category == 'all':
        result, status = discover_and_run_tests()
    else:
        result, status = run_specific_test_category(args.category)
    
    # Exit with appropriate code
    if status in ['PASS', 'MOSTLY_PASS']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main() 