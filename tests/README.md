# Test Suite Documentation

This directory contains tests for the Inverting-TPR project. The test suite is organized into **basic tests** (run by default) and **comprehensive tests** (run on demand).

## Test Structure

### Basic Tests (Default Mode)
- **Purpose**: Test core functionality of each script with minimal configurations
- **Execution Time**: Fast (~1 minute)
- **Coverage**: One configuration per script to ensure basic functionality works
- **Run Command**: `uv run -m pytest/ -m "not comprehensive"`

### Comprehensive Tests (On-Demand Mode)  
- **Purpose**: Test multiple parameter combinations and edge cases
- **Execution Time**: Slower (~10-30 minutes depending on combinations)
- **Coverage**: Multiple tasks, architectures, role schemes, regularization methods, etc.
- **Run Command**: `uv run -m pytest/ -m comprehensive`

## Test Categories

### Script Integration Tests (`test_script_integration.py`)

#### Basic Tests (Always Run)
- `TestBasicDigitsWorkflow`: Basic digit generation, training, and inversion
- `TestBasicSentenceWorkflow`: Basic sentence generation and training  
- `TestBasicIntegrationWorkflows`: End-to-end workflow tests

#### Comprehensive Tests (On-Demand)
- `TestDigitTaskVariations`: Tests 4 different transformation tasks (copy, reverse, sort_ascending, interleave)
- `TestArchitectureRoleVariations`: Tests 3 architectures × 3 role schemes = 9 combinations
- `TestRegularizationVariations`: Tests 3 regularization methods (l2, atol, topk)
- `TestSentenceRoleSchemeVariations`: Tests different sentence role schemes (svo, bow)
- `TestProbeComponentVariations`: Tests different probe skip combinations
- `TestConfigurationRobustness`: Edge cases and minimal data sizes

## Configuration Management

Test configurations are generated dynamically using `config_generator.py`:

- **`TestConfigGenerator`**: Creates gin configs for different parameter combinations
- **`COMPREHENSIVE_TEST_PARAMS`**: Defines parameter sets for comprehensive testing
- **Benefits**: Separates config logic from test logic, makes adding new configurations easy

## Running Tests

```bash
# Default: Run basic tests only (fast)
uv run -m pytest tests/test_script_integration.py

# Run basic tests including slow ones
uv run -m pytest tests/test_script_integration.py -m "not comprehensive"

# Run comprehensive tests only
uv run -m pytest tests/test_script_integration.py -m comprehensive

# Run only fast basic tests (exclude slow)  
uv run -m pytest tests/test_script_integration.py -m "not slow and not comprehensive"

# Run all tests (basic + comprehensive + slow)
uv run -m pytest tests/test_script_integration.py -m ""
```

## Adding New Tests

### Adding a Basic Test
1. Add test method to appropriate `TestBasic*` class
2. Use `config_gen` fixture to generate configurations
3. Test should run quickly with minimal data

### Adding a Comprehensive Test
1. Add test method to appropriate comprehensive test class
2. Add `@pytest.mark.comprehensive` decorator
3. Use `@pytest.mark.parametrize` for parameter variations
4. Add new parameter sets to `COMPREHENSIVE_TEST_PARAMS` in `config_generator.py`

### Adding New Configurations
1. Add new method to `TestConfigGenerator` class
2. Update `COMPREHENSIVE_TEST_PARAMS` if needed for parametrized tests
3. Ensure configs use minimal settings for test speed

## Test Markers

- `@pytest.mark.slow`: Tests that take longer than usual (included in default but can be skipped)
- `@pytest.mark.comprehensive`: Comprehensive configuration tests (excluded by default)

## Best Practices

1. **Basic tests** should be fast and test core functionality
2. **Comprehensive tests** should test parameter variations and edge cases  
3. Use **minimal data sizes** in all tests (small vocab, short sequences, few epochs)
4. **Separate config generation** from test logic using `config_generator.py`
5. **Document expected behavior** in test docstrings
6. **Use descriptive test names** that indicate what configuration is being tested 