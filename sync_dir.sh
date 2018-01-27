#!/bin/bash
# element laptop
#rsync --delete -avz /home/david/git_repos/pixelsnail-public/ david.krueger@8gpu03.elementai.lan:/mnt/AIDATA/home/david.krueger/dev/pixelsnail-public
# macbook
rsync --delete -avz /Users/david/pixelsnail-public/ david.krueger@8gpu03.elementai.lan:/mnt/AIDATA/home/david.krueger/dev/pixelsnail-public
