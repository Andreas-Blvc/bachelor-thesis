#!/bin/bash

cd /var/www/html/bachelor-thesis
week_ctr=$(cat presentations/data/week_counter.txt)
git add .
git commit -m "week $week_ctr"
git push
git subtree push --prefix presentations origin gh-pages