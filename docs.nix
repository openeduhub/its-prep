{
  runCommand,
  writeShellScript,
  sphinx,
  sphinx-rtd-theme,
  sphinx-autodoc-typehints,
  its-prep,
}:
runCommand "docs"
  {
    buildInputs = [
      sphinx
      sphinx-rtd-theme
      sphinx-autodoc-typehints
      (its-prep.overridePythonAttrs { doCheck = false; })
    ];
  }
  (
    writeShellScript "docs.sh" ''
      sphinx-build -b html ${./docs} $out
    ''
  )
