{ mkShell, python3 }:
mkShell {
  packages = [
    (python3.withPackages (
      py-pkgs: with py-pkgs; [
        black
        flake8
        isort
        ipython
        mypy
        pytest
        pytest-cov
        hypothesis
        debugpy
      ] ++ py-pkgs.its-prep.dependencies
    ))
  ];
}
