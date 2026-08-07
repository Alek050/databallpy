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
	$(call echotask,"formatlint","formats and lints code using ruff")
	$(call echotask,"test","runs all tests")
	$(call echotask,"docs","runs sphinx code to create docs")
	$(call echotask,"release-prep","bumps version from CHANGELOG.md and commits/tags on develop")
	$(call echotask,"publish","builds and publishes to PyPI on main after release PR merge")
	@echo

deps:
	poetry install --all-extras --with docs,developing

format:
	poetry run ruff format $(LINT_FILES)

formatcheck:
	poetry run ruff format --check $(LINT_FILES)

ruff_fix: ## Run ruff lint check with auto fix
	poetry run ruff check --fix $(LINT_FILES)

formatlint: format ruff_fix

test:
	$(PYTEST) $(PYTEST_ARGS_COV)

documentation:
	poetry run jupyter-book build docs/ --all -W

ready: formatlint test documentation

release-prep:
	@if ! git diff --quiet || ! git diff --cached --quiet; then \
		echo "Working tree not clean - commit or stash changes first."; \
		exit 1; \
	fi
	@LINE=$$(grep -m1 -E '^### V[0-9]+\.[0-9]+\.[0-9]+' CHANGELOG.md); \
	if [ -z "$$LINE" ]; then \
		echo "No '### V<version>' heading found in CHANGELOG.md."; \
		exit 1; \
	fi; \
	if echo "$$LINE" | grep -q '('; then \
		echo "Top CHANGELOG.md entry is already dated - draft the next version's section first."; \
		exit 1; \
	fi; \
	VERSION=$$(echo "$$LINE" | sed -E 's/^### V([0-9]+\.[0-9]+\.[0-9]+).*/\1/'); \
	echo "Bumping to $$VERSION"; \
	poetry version "$$VERSION"; \
	sed -i.bak -E "s/__version__ == \"[0-9]+\.[0-9]+\.[0-9]+\"/__version__ == \"$$VERSION\"/" tests/test_databallpy.py && rm -f tests/test_databallpy.py.bak; \
	TODAY=$$(date +%d/%m/%Y); \
	sed -i.bak -E "s|^### V$${VERSION}\$$|### V$${VERSION} ($${TODAY})|" CHANGELOG.md && rm -f CHANGELOG.md.bak; \
	git add pyproject.toml tests/test_databallpy.py CHANGELOG.md; \
	git commit -m "Bump version to $$VERSION"; \
	git tag "v$$VERSION"; \
	echo; \
	echo "Done. Next steps:"; \
	echo "  1. git push && git push origin v$$VERSION"; \
	echo "  2. Open a PR from develop into main"; \
	echo "  3. After merge, on main: make publish"

publish:
	poetry build
	poetry publish
