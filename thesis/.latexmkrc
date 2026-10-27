# Bachelorarbeit: pdflatex + biber, Hilfsdateien in build/, PDF im Hauptordner
$pdf_mode = 1;
$bibtex_use = 2;
$aux_dir = 'build';
$out_dir = '.';
$emulate_aux = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
@default_files = ('main.tex');
