{
  description = "Dependency and Build Process for the Text Pre-Processing Pipeline";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    let
      nix-filter = self.inputs.nix-filter.lib;

      ### declare the python packages used for building, docs & development
      python-packages-build = py-pkgs:
        with py-pkgs; [
          numpy
          # NLP
          spacy
          spacy_models.de_core_news_lg
          # language detection, also used in trafilatura
          py3langid
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
          # unit tests
          pytest
          pytest-cov
          hypothesis
          # debugger
          debugpy
        ]
        ++ (python-packages-build py-pkgs)
        ++ (python-packages-docs py-pkgs);

      ### declare how the python package shall be built
      nlprep-lib = py-pkgs: py-pkgs.buildPythonPackage rec {
        pname = "nlprep";
        version = "0.1.4";
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
    in
    {
      lib = {
        nlprep = nlprep-lib;
      };
      overlays.default = (final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (python-final: python-prev: {
            data-utils = self.outputs.lib.data-utils python-final;
          })
        ];
      });
    } // flake-utils.lib.eachDefaultSystem (system:
      let
        # import the packages from nixpkgs
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
      in
      {
        packages = rec {
          default = nlprep;
          nlprep = nlprep-lib python.pkgs;
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
