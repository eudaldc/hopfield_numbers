# Keeps builds consistent when latexmk is started from this folder (e.g. LaTeX Workshop).
$do_cd = 1;

# Cross-document references via xr-hyper require both .aux files to be in sync.
@default_files = ('main.tex', 'supplementary_material.tex');

add_cus_dep('main.aux', 'supplementary_material.pdf', 0, 'noop');
add_cus_dep('supplementary_material.aux', 'main.pdf', 0, 'noop');
sub noop { return 0; }
