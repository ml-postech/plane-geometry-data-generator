splits=( train valid test )
n_cnt=( 50000 10000 10000 )
n_workers=50
root_path="./data"

for i in {0..2}
do
    split=${splits[i]}
    n_problems=${n_cnt[i]}
    python benchmark_cyclic.py --n_problems ${n_problems} --split ${split} --image_folder "${root_path}/ConcyclicBenchmark/images" --out_file "${root_path}/ConcyclicBenchmark/${split}/problems.jsonl" --n_workers ${n_workers}
done


for i in {0..2}
do
    split=${splits[i]}
    n_problems=${n_cnt[i]}
    python benchmark_object.py --n_problems ${n_problems} --split ${split} --image_folder "${root_path}/ObjectBenchmark/images" --out_file "${root_path}/ObjectBenchmark/${split}/problems.jsonl" --n_workers ${n_workers}
done

for i in {0..2}
do
    split=${splits[i]}
    n_problems=${n_cnt[i]}
    python benchmark_twolines.py --n_problems ${n_problems} --split ${split} --image_folder "${root_path}/TwoLinesBenchmark/images" --out_file "${root_path}/TwoLinesBenchmark/${split}/problems.jsonl" --n_workers ${n_workers}
done

for i in {0..2}
do
    split=${splits[i]}
    n_problems=${n_cnt[i]}
    python benchmark_square.py --n_problems ${n_problems} --split ${split} --image_folder "${root_path}/SquareShapeBenchmark/images" --out_file "${root_path}/SquareShapeBenchmark/${split}/problems.jsonl" --n_workers ${n_workers}
done

for i in {0..2}
do
    split=${splits[i]}
    n_problems=${n_cnt[i]}
    python benchmark_angledetection.py --n_problems ${n_problems} --split ${split} --image_folder "${root_path}/AngleDetectionBenchmark/images" --out_file "${root_path}/AngleDetectionBenchmark/${split}/problems.jsonl" --n_workers ${n_workers}
done
