LINT_FILES = tests/ databallpy/
TEST_PATH = tests/
PYTEST = poetry run python -m pytest $(TEST_PATH) --pythonwarnings=once
PYTEST_ARGS_COV = --cov-report=term-missing --cov-report=xml --cov=databallpy --cov-config=.coveragerc

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
	$(call echotask,"format","formats code using ruff")
	$(call echotask,"formatcheck","checks format using ruff")
	$(call echotask,"lint","lints all code using ruff")
	$(call echotask,"formatlint","formats and lints code using ruff")
	$(call echotask,"test","runs all tests")
	$(call echotask,"docs","runs sphinx code to create docs")
	@echo

deps:
	poetry install --with docs,developing

format:
	poetry run ruff format $(LINT_FILES)

formatcheck:
	poetry run ruff format --check $(LINT_FILES)

ruff_fix: ## Run ruff lint check with auto fix
	poetry run ruff check --fix $(LINT_FILES)

ruff: ## Run ruff lint checs
	poetry run ruff check $(LINT_FILES)

lint: formatcheck ruff

formatlint: ruff format $(LINT_FILES)

test:
	$(PYTEST) $(PYTEST_ARGS_COV)

documentation: 
	poetry run jupyter-book build docs/ --all -W

ready: formatlint test documentation
