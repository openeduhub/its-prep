{
  buildPythonPackage,
  pytestCheckHook,
  hypothesis,
  nix-filter,
  numpy,
  spacy,
  spacy_models,
  py3langid,

}:
buildPythonPackage {
  pname = "its-prep";
  version = "0.1.4";
  # only include the package-related files
  src = nix-filter {
    root = ./.;
    include = [
      "its_prep"
      "test"
      ./setup.py
      ./requirements.txt
    ];
    exclude = [ (nix-filter.matchExt "pyc") ];
  };
  propagatedBuildInputs = [
    numpy
    spacy
    spacy_models.de_core_news_lg
    py3langid
  ];
  # use pytestCheckHook to run pytest after building
  nativeCheckInputs = [
    pytestCheckHook
    hypothesis
  ];
  # use the hypothesis profile that is more reproducible
  pytestFlagsArray = [ "--hypothesis-profile=build" ];
}
