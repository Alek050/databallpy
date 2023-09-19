LINT_FILES = tests/ databallpy/
TEST_PATH = tests/
PYTEST = poetry run py.test $(TEST_PATH) --pythonwarnings=once
PYTEST_ARGS_COV = --cov-report=term-missing --cov-report=html --cov=databallpy --cov-config=.coveragerc


define echotask
	@tput setaf 6
	@echo -n "  $1"
	@tput setaf 8
	@echo -n " - "
	@tput sgr0
	@echo $2
endef

help:
	@echo
	$(call echotask,"deps","installs and updates all dependencies for developing")
	$(call echotask,"format","formats code using isort and black")
	$(call echotask,"formatcheck","checks format using isort and black")
	$(call echotask,"flake8","lints code using flake8")
	$(call echotask,"lint","lints all code using flake8 isort and black")
	$(call echotask,"formatlint","formats and lints code using flake8 isort and black")
	$(call echotask,"test","runs all tests")
	$(call echotask,"docs","runs sphinx code to create docs")
	@echo

deps_poetry:
	pip install --upgrade poetry

deps: deps_poetry
	poetry install --with docs

format: deps_poetry
	poetry run isort --filter-files $(LINT_FILES)
	poetry run black $(LINT_FILES)

formatcheck: deps_poetry
	poetry run isort --check-only --filter-files $(LINT_FILES)
	poetry run black --check $(LINT_FILES)

flake8: deps_poetry
	poetry run flake8 $(LINT_FILES)

lint: flake8 formatcheck

formatlint: format flake8

test: deps_poetry
	$(PYTEST) $(PYTEST_ARGS_COV)

documentation: deps_poetry
	poetry run make html --directory docs/
