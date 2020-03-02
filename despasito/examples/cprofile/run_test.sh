#! /bin/bash

IFS=$'\n\n'
set -f

S=("--jit" "--cython" " ")
P=("../CO2_H2O/input_liquid.json" "../CO2_H2O/input_xi.json" "../CO2_H2O/input_xi.json" "../butane_solubility/input_xi.json" "../hexane_heptane/input_xi.json" "../propylacetate_cyclohexane/input_paper.json")

for s in ${S[@]}; do
    for i in ${!P[@]}; do
        echo "time python -m cProfile -o output_${i}${s} -m despasito -i ${P[$i]} ${s}"
        time python -m cProfile -o output_${i}${s}.txt -m despasito -i ${P[$i]} ${s}
    done    
done

