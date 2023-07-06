{
  description = "Dependency and Build Process for the Text Pre-Processing Pipeline";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let

        pkgs = nixpkgs.legacyPackages.${system};

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
            spacy
          ];
        python-build = pkgs.python3.withPackages python-packages-build;

        python-packages-test = python-packages:
          with python-packages; [
            # type checking
            mypy
            # unit testing
            pytest
            pytest-cov
            hypothesis
          ] ++ (python-packages-build python-packages);
        python-test = pkgs.python3.withPackages python-packages-build;

        python-packages-devel = python-packages:
          with python-packages; [
            # coding utilities
            black
            flake8
            isort
            ipython
          ] ++ (python-packages-test python-packages);
        python-devel = pkgs.python3.withPackages python-packages-devel;

        # declare how the python package shall be built
        nlprep = python-build.pkgs.buildPythonPackage {
          pname = "nlprep";
          version = "0.0.1";

          propagatedBuildInputs = (python-packages-build python-build.pkgs);
          nativeCheckInputs = (python-packages-test python-test.pkgs);

          src = ./.;
        };

      in {
        defaultPackage = nlprep;
        devShell = pkgs.mkShell {
          buildInputs = [
            python-devel
            # python language server
            pkgs.nodePackages.pyright
          ];
        };
      }
    );}
