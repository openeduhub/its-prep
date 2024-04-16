{
  description = "Dependency and Build Process for the Text Pre-Processing Pipeline";

  inputs = {
    # track unstable because one of our dependencies, py3langid, is only
    # available in unstable. change this to nixos-24.05, once that is
    # available.
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    {
      overlays = import ./overlays.nix {
        inherit (nixpkgs) lib;
        nix-filter = self.inputs.nix-filter.lib;
      };
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        # import the packages from nixpkgs
        pkgs = nixpkgs.legacyPackages.${system}.extend self.outputs.overlays.default;
      in
      {
        packages = rec {
          default = its-prep;
          its-prep = pkgs.python3Packages.its-prep;
          docs = pkgs.python3Packages.callPackage ./docs.nix { };
        };
        devShells.default = pkgs.callPackage ./shell.nix { };
      }
    );
}
