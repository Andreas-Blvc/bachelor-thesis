#!/bin/bash

cd /var/www/html/bachelor-thesis/
week_ctr=$(cat presentations/data/week_counter.txt)
week_ctr=$((week_ctr - 1))
git add presentations
git commit -m "week $week_ctr"
git push
git subtree push --prefix presentations origin gh-pages