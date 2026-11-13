from ekg import check_env, paths


def test_project_root_contains_pyproject():
    assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()


def test_imports():
    assert check_env.check_imports()


def test_training_step():
    assert check_env.check_training_step()
