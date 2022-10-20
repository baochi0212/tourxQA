data_var = data external interim processed raw
report_var = report figures
src_var = src dataset utils models visualization 
#have to use all for multiple targets
all: data_var report_var src_var docs 
data_var: $(data_var)
	echo "finished data"
report_var: $(report_var)
	echo "finished report"
src_var: $(src_var)
	echo "finished src"
data: 
	mkdir data
external:
	mkdir data/external
interim:
	mkdir data/interim
processed:
	mkdir data/processed
raw:
	mkdir data/raw
report:
	mkdir report
figures:
	mkdir report/figures
src:
	mkdir src
dataset:
	mkdir src/dataset
utils:
	mkdir src/utils
models:
	mkdir src/models
visualization:
	mkdir src/runtime
	touch src/visualization.ipynb
docs: 
	mkdir docs


	
	