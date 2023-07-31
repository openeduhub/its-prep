#!/bin/sh
mkdir html
nix build .\#docs --out-link docs-result &&
    cp -rf $(readlink -f docs-result)/html/* html &&
    rm docs-result &&
    chmod -R 755 html
