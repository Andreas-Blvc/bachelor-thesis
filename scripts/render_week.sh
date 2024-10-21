cd resources

mv presentation_template.html ../..

node render-html.js $1

sed -i 's/<!--\* //g' rendered-page.html
sed -i 's/ \*-->//g' rendered-page.html
sed -i 's/\/\/\* //g' rendered-page.html

mv ../../presentation_template.html .
mv rendered-page.html ../..
