rounds.csv: download.py
	python download.py --shape RD > rounds.csv

diamonds.html: diamonds.Rmd rounds.csv
	Rscript -e "rmarkdown::render('diamonds.Rmd')"

BLOG = ~/Dropbox/amarder.github.io
IMG_FOLDER = $(BLOG)/static/images/diamonds
IMG_URL = /images/diamonds

blog: diamonds.html
	rm -r $(IMG_FOLDER)
	cp -r figure $(IMG_FOLDER)
	cat diamonds.md | sed "s+(figure/+($(IMG_URL)/+" | sed "s/\_/\\\_/g" > $(BLOG)/content/post/diamonds.md

diamonds.docx: diamonds.html
	pandoc diamonds.md -o diamonds.docx
