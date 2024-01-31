{
  description = "Dependency and Build Process for the Text Pre-Processing Pipeline";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    let
      nix-filter = self.inputs.nix-filter.lib;

      # build the spaCy language processing pipeline as a python package
      de_dep_news_trf = py-pkgs: py-pkgs.buildPythonPackage rec {
        pname = "de_core_news_lg";
        version =
          if "3.8" > py-pkgs.spacy.version && py-pkgs.spacy.version >= "3.7" then "3.7.0" else
          if "3.6" > py-pkgs.spacy.version && py-pkgs.spacy.version >= "3.5" then "3.5.0" else
          builtins.throw "Unsupported spacy version";
        src = builtins.fetchTarball {
          url = "https://github.com/explosion/spacy-models/releases/download/${pname}-${version}/${pname}-${version}.tar.gz";
          sha256 =
            if "3.8" > py-pkgs.spacy.version && py-pkgs.spacy.version >= "3.7" then "oksQXT/QbUno4y0l5t04FckWNULylLtX9spvgBsaNR0=" else
            if "3.6" > py-pkgs.spacy.version && py-pkgs.spacy.version >= "3.5" then "oOrxOoe+SyleTsDO9WYB25Vvs4LX6B4aJPlGbMRsAk4=" else
            builtins.throw "Unsupported spacy version";
        };
        doCheck = false;
        propagatedBuildInputs = with py-pkgs; [ spacy ];
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
