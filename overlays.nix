{ lib, nix-filter }:
rec {
  default = its-prep;

  its-prep = (
    final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (python-final: python-prev: {
          its-prep = python-final.callPackage ./python-lib.nix { inherit nix-filter; };
        })
      ];
    }
  );
}
