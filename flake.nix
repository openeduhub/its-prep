{
  description = "Dependency and Build Process for the Text Pre-Processing Pipeline";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python310;
        nix-filter = self.inputs.nix-filter.lib;

        # build the spaCy language processing pipeline as a python package
        de_dep_news_trf = py-pkgs: py-pkgs.buildPythonPackage rec {
          pname = "de_core_news_lg";
          version = "3.7.0";
          src = pkgs.fetchzip {
            url = "https://github.com/explosion/spacy-models/releases/download/${pname}-${version}/${pname}-${version}.tar.gz";
            hash = "sha256-oksQXT/QbUno4y0l5t04FckWNULylLtX9spvgBsaNR0=";
          };
          doCheck = false;
          propagatedBuildInputs = with py-pkgs; [
            spacy
            # spacy-transformers
          ];
        };

        ### declare the python packages used for building, docs & development
        python-packages-build = py-pkgs:
          with py-pkgs; [
            (de_dep_news_trf py-pkgs)
            numpy
            spacy
          ];

        python-packages-docs = py-pkgs:
          with py-pkgs; [
            sphinx
            sphinx-rtd-theme
            sphinx-autodoc-typehints
          ];

        python-packages-devel = py-pkgs:
          with py-pkgs; [
            # coding utilities
            black
            flake8
            isort
            ipython
            # type checking
            mypy
            # writing tests
            pytest
            pytest-cov
            hypothesis
          ]
          ++ (python-packages-build py-pkgs)
          ++ (python-packages-docs py-pkgs);

        ### declare how the python package shall be built
        nlprep-lib = py-pkgs: py-pkgs.buildPythonPackage rec {
          pname = "nlprep";
          version = "0.1.3";
          # only include the package-related files
          src = nix-filter {
            root = self;
            include = [
              "${pname}"
              "test"
              ./setup.py
              ./requirements.txt
            ];
            exclude = [ (nix-filter.matchExt "pyc") ];
          };
          propagatedBuildInputs = (python-packages-build py-pkgs);
          # use pytestCheckHook to run pytest after building
          nativeCheckInputs = with py-pkgs; [
            pytestCheckHook
            hypothesis
          ];
        };
        nlprep = nlprep-lib python.pkgs;

        ### declare build system for the documentation
        docs = pkgs.runCommand "docs"
          {
            buildInputs = [
              (python-packages-docs python.pkgs)
              (nlprep.override { doCheck = false; })
            ];
          }
          (pkgs.writeShellScript "docs.sh" ''
            sphinx-build -b html ${./docs} $out
          '');
      in
      {
        packages = {
          inherit nlprep docs;
          default = nlprep;
        };
        lib = {
          nlprep = nlprep-lib;
        };
        devShells.default = pkgs.mkShell {
          buildInputs = [
            (python.withPackages python-packages-devel)
            # python language server
            pkgs.nodePackages.pyright
          ];
        };
      }
    );
}
