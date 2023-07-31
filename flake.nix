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
        pkgs = import nixpkgs {inherit system;};
        nix-filter = self.inputs.nix-filter.lib;
        python = pkgs.python310;

        # build the spaCy language processing pipeline as a python package
        de_dep_news_trf = with pkgs.python3Packages;
          buildPythonPackage rec {
            pname = "de_dep_news_trf";
            version = "3.5.0";
            src = pkgs.fetchzip {
              url = "https://github.com/explosion/spacy-models/releases/download/${pname}-${version}/${pname}-${version}.tar.gz";
              hash = "sha256-MleftGMj+VK8hAE+EOg0VhtFUg/oOH3grPbRzklyiVE=";
            };
            doCheck = false;
            propagatedBuildInputs = [
              spacy
              spacy-transformers
            ];
          };

        # declare the python packages used for building, testing & developing
        python-packages-build = python-packages:
          with python-packages; [
            # NLP
            de_dep_news_trf
            numpy
            spacy
          ];
        python-build = python.withPackages python-packages-build;

        python-packages-docs = python-packages:
          with python-packages; [
            sphinx
            sphinx-rtd-theme
            sphinx-autodoc-typehints
          ];
        python-docs = python.withPackages python-packages-docs;

        python-packages-devel = python-packages:
          with python-packages; [
            # coding utilities
            black
            flake8
            isort
            ipython
            # type checking
            mypy
            # unit testing
            pytest
            pytest-cov
            hypothesis
          ]
          ++ (python-packages-build python-packages)
          ++ (python-packages-docs python-packages);
        python-devel = python.withPackages python-packages-devel;

        # declare how the python package shall be built
        nlprep = with python-build.pkgs; buildPythonPackage rec {
          pname = "nlprep";
          version = "0.1.2";
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
          propagatedBuildInputs = (python-packages-build python-build.pkgs);
          nativeCheckInputs = [ pytestCheckHook hypothesis ];
        };

        docs = pkgs.runCommand "docs" {
          buildInputs = [
            python-docs
            (nlprep.override {doCheck = false;})
          ];
        } (pkgs.writeShellScript "docs.sh" ''
            sphinx-build -M html ${./docs} $out
          ''); 

      in {
        packages = rec {
          inherit nlprep docs;
          default = nlprep;
        };
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python-devel
            # python language server
            pkgs.nodePackages.pyright
          ];
        };
      }
    );}
